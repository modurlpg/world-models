import os, sys
import numpy as np
from functools import partial
import torch
import torch.utils.data
from torch.nn import functional as F
from torch import optim
from torch.distributions.normal import Normal
from torch.multiprocessing import Process, Queue
from torchvision import transforms
from torchvision.utils import save_image
import data.carracing as carracing
from tqdm import tqdm
import cma
from time import sleep

from models.vae import VAE
from models.mdrnn import MDRNN, MDRNNCell
from models.controller import Controller

from data.loaders import RolloutObservationDataset, RolloutSequenceDataset

from utils.misc import LSIZE, ASIZE, RSIZE, RED_SIZE, SIZE
from utils.misc import save_checkpoint
from utils.misc import RolloutGenerator
from utils.misc import load_parameters
from utils.misc import flatten_parameters


## WARNING : THIS SHOULD BE REPLACE WITH PYTORCH 0.5
from utils.learning import EarlyStopping
from utils.learning import ReduceLROnPlateau


cuda = torch.cuda.is_available()
device = torch.device("cuda" if cuda else "cpu")
torch.manual_seed(123)

#temporarily hard-code dirs
rollout_root_dir = './data/rollout/carracing'
random_rollout_dir = './data/rollout/carracing/random'
random_rollout_num = 10

vae_dir = './data/vae'
V_BATCH_SIZE = 32

rnn_dir = './data/mdrnn'
M_BATCH_SIZE = 16
M_SEQ_LEN = 32

ctrl_dir = './data/ctrl'
tmp_dir = './data/tmp'
log_dir = './data/log'
C_N_SAMPLES = 4
C_POP_SIZE = 4  # pipaek : for single process
rollout_time_limit = 1000


def generate_random_rollout_data(rollout_dir, random_rollout_num):
    if not os.path.exists(rollout_dir):
        os.makedirs(rollout_dir)

    carracing.generate_data(random_rollout_num, rollout_dir, "brown")


def make_vae_dataset(rollout_dir):
    transform_train = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((RED_SIZE, RED_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    transform_test = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((RED_SIZE, RED_SIZE)),
        transforms.ToTensor(),
    ])
    dataset_train = RolloutObservationDataset(rollout_dir, transform_train, train=True)
    dataset_test = RolloutObservationDataset(rollout_dir, transform_test, train=False)

    return dataset_train, dataset_test


def make_mdrnn_dataset(rollout_dir):
    transform = transforms.Lambda(
        lambda x: np.transpose(x, (0, 3, 1, 2)) / 255)

    dataset_train = RolloutSequenceDataset(rollout_dir, M_SEQ_LEN, transform, train=True, buffer_size=30)
    dataset_test = RolloutSequenceDataset(rollout_dir, M_SEQ_LEN, transform, train=False, buffer_size=10)

    return dataset_train, dataset_test


# Reconstruction + KL divergence losses summed over all elements and batch
def v_loss_function(recon_x, x, mu, logsigma):
    """ VAE loss function """
    BCE = F.mse_loss(recon_x, x, size_average=False)

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + 2 * logsigma - mu.pow(2) - (2 * logsigma).exp())
    return BCE + KLD


def vae_train(epoch, model, dataset, train_loader, optimizer):
    """ One training epoch """
    model.train()
    dataset.load_next_buffer()
    train_loss = 0
    for batch_idx, data in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = v_loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % 20 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item() / len(data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
        epoch, train_loss / len(train_loader.dataset)))


def vae_test(model, dataset, test_loader):
    """ One test epoch """
    model.eval()
    dataset.load_next_buffer()
    test_loss = 0
    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            recon_batch, mu, logvar = model(data)
            test_loss += v_loss_function(recon_batch, data, mu, logvar).item()

    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))
    return test_loss


def v_model_train_proc(vae_dir, model, dataset_train, dataset_test, optimizer, scheduler, earlystopping, skip_train=False, max_train_epochs=1000):

    train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=V_BATCH_SIZE, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=V_BATCH_SIZE, shuffle=True)

    # check vae dir exists, if not, create it
    if not os.path.exists(vae_dir):
        os.mkdir(vae_dir)

    # sample image dir for each epoch
    sample_dir = os.path.join(vae_dir, 'samples')
    if not os.path.exists(sample_dir):
        os.mkdir(sample_dir)


    reload_file = os.path.join(vae_dir, 'best.tar')
    if os.path.exists(reload_file):
        state = torch.load(reload_file)
        print("Reloading model at epoch {}"
              ", with test error {}".format(
            state['epoch'],
            state['precision']))
        v_model.load_state_dict(state['state_dict'])
        optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler'])
        earlystopping.load_state_dict(state['earlystopping'])

    if skip_train:
        return   # pipaek : 트레이닝을 통한 모델 개선을 skip하고 싶을 때..

    cur_best = None

    for epoch in range(1, max_train_epochs + 1):
        vae_train(epoch, model, dataset_train, train_loader, optimizer)
        test_loss = vae_test(model, dataset_test, test_loader)
        scheduler.step(test_loss)
        earlystopping.step(test_loss)

        # checkpointing
        best_filename = os.path.join(vae_dir, 'best.tar')
        filename = os.path.join(vae_dir, 'checkpoint.tar')
        is_best = not cur_best or test_loss < cur_best
        if is_best:
            cur_best = test_loss

        save_checkpoint({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'precision': test_loss,
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'earlystopping': earlystopping.state_dict()
        }, is_best, filename, best_filename)

        #if not args.nosamples:
        with torch.no_grad():
            sample = torch.randn(RED_SIZE, LSIZE).to(device)
            sample = model.decoder(sample).cpu()
            save_image(sample.view(64, 3, RED_SIZE, RED_SIZE),
                       os.path.join(sample_dir, 'sample_' + str(epoch) + '.png'))

        if earlystopping.stop:
            print("End of Training because of early stopping at epoch {}".format(epoch))
            break


def m_model_train_proc(rnn_dir, model, v_model, dataset_train, dataset_test, optimizer, scheduler, earlystopping, skip_train=False, max_train_epochs=50):

    train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=M_BATCH_SIZE, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=M_BATCH_SIZE)

    # check rnn dir exists, if not, create it
    if not os.path.exists(rnn_dir):
        os.mkdir(rnn_dir)

    rnn_file = os.path.join(rnn_dir, 'best.tar')
    if os.path.exists(rnn_file):
        state = torch.load(rnn_file)
        print("Reloading model at epoch {}"
              ", with test error {}".format(
            state['epoch'],
            state['precision']))
        model.load_state_dict(state['state_dict'])
        optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler'])
        earlystopping.load_state_dict(state['earlystopping'])

    if skip_train:
        return   # pipaek : 트레이닝을 통한 모델 개선을 skip하고 싶을 때..

    def data_pass(epoch, train):  # pylint: disable=too-many-locals
        """ One pass through the data """
        if train:
            model.train()
            loader = train_loader
        else:
            model.eval()
            loader = test_loader

        loader.dataset.load_next_buffer()

        cum_loss = 0
        cum_gmm = 0
        cum_bce = 0
        cum_mse = 0

        pbar = tqdm(total=len(loader.dataset), desc="Epoch {}".format(epoch))
        for i, data in enumerate(loader):
            obs, action, reward, terminal, next_obs = [arr.to(device) for arr in data]

            # transform obs
            latent_obs, latent_next_obs = to_latent(obs, next_obs)

            if train:
                losses = get_loss(latent_obs, action, reward,
                                  terminal, latent_next_obs)

                optimizer.zero_grad()
                losses['loss'].backward()
                optimizer.step()
            else:
                with torch.no_grad():
                    losses = get_loss(latent_obs, action, reward,
                                      terminal, latent_next_obs)

            cum_loss += losses['loss'].item()
            cum_gmm += losses['gmm'].item()
            cum_bce += losses['bce'].item()
            cum_mse += losses['mse'].item()

            pbar.set_postfix_str("loss={loss:10.6f} bce={bce:10.6f} "
                                 "gmm={gmm:10.6f} mse={mse:10.6f}".format(
                loss=cum_loss / (i + 1), bce=cum_bce / (i + 1),
                gmm=cum_gmm / LSIZE / (i + 1), mse=cum_mse / (i + 1)))
            pbar.update(M_BATCH_SIZE)
        pbar.close()
        return cum_loss * M_BATCH_SIZE / len(loader.dataset)

    def to_latent(obs, next_obs):
        """ Transform observations to latent space.

        :args obs: 5D torch tensor (BSIZE, SEQ_LEN, ASIZE, SIZE, SIZE)
        :args next_obs: 5D torch tensor (BSIZE, SEQ_LEN, ASIZE, SIZE, SIZE)

        :returns: (latent_obs, latent_next_obs)
            - latent_obs: 4D torch tensor (BSIZE, SEQ_LEN, LSIZE)
            - next_latent_obs: 4D torch tensor (BSIZE, SEQ_LEN, LSIZE)
        """
        with torch.no_grad():
            obs, next_obs = [
                F.upsample(x.view(-1, 3, SIZE, SIZE), size=RED_SIZE,
                           mode='bilinear', align_corners=True)
                for x in (obs, next_obs)]

            (obs_mu, obs_logsigma), (next_obs_mu, next_obs_logsigma) = [
                v_model(x)[1:] for x in (obs, next_obs)]

            latent_obs, latent_next_obs = [
                (x_mu + x_logsigma.exp() * torch.randn_like(x_mu)).view(M_BATCH_SIZE, M_SEQ_LEN, LSIZE)
                for x_mu, x_logsigma in
                [(obs_mu, obs_logsigma), (next_obs_mu, next_obs_logsigma)]]
        return latent_obs, latent_next_obs

    def get_loss(latent_obs, action, reward, terminal, latent_next_obs):
        """ Compute losses.

        The loss that is computed is:
        (GMMLoss(latent_next_obs, GMMPredicted) + MSE(reward, predicted_reward) +
             BCE(terminal, logit_terminal)) / (LSIZE + 2)
        The LSIZE + 2 factor is here to counteract the fact that the GMMLoss scales
        approximately linearily with LSIZE. All losses are averaged both on the
        batch and the sequence dimensions (the two first dimensions).

        :args latent_obs: (BSIZE, SEQ_LEN, LSIZE) torch tensor
        :args action: (BSIZE, SEQ_LEN, ASIZE) torch tensor
        :args reward: (BSIZE, SEQ_LEN) torch tensor
        :args latent_next_obs: (BSIZE, SEQ_LEN, LSIZE) torch tensor

        :returns: dictionary of losses, containing the gmm, the mse, the bce and
            the averaged loss.
        """
        latent_obs, action, \
        reward, terminal, \
        latent_next_obs = [arr.transpose(1, 0)
                           for arr in [latent_obs, action,
                                       reward, terminal,
                                       latent_next_obs]]
        mus, sigmas, logpi, rs, ds = model(action, latent_obs)
        gmm = gmm_loss(latent_next_obs, mus, sigmas, logpi)
        bce = F.binary_cross_entropy_with_logits(ds, terminal)
        mse = F.mse_loss(rs, reward)
        loss = (gmm + bce + mse) / (LSIZE + 2)
        return dict(gmm=gmm, bce=bce, mse=mse, loss=loss)


    def gmm_loss(batch, mus, sigmas, logpi, reduce=True):  # pylint: disable=too-many-arguments
        """ Computes the gmm loss.

        Compute minus the log probability of batch under the GMM model described
        by mus, sigmas, pi. Precisely, with bs1, bs2, ... the sizes of the batch
        dimensions (several batch dimension are useful when you have both a batch
        axis and a time step axis), gs the number of mixtures and fs the number of
        features.

        :args batch: (bs1, bs2, *, fs) torch tensor
        :args mus: (bs1, bs2, *, gs, fs) torch tensor
        :args sigmas: (bs1, bs2, *, gs, fs) torch tensor
        :args logpi: (bs1, bs2, *, gs) torch tensor
        :args reduce: if not reduce, the mean in the following formula is ommited

        :returns:
        loss(batch) = - mean_{i1=0..bs1, i2=0..bs2, ...} log(
            sum_{k=1..gs} pi[i1, i2, ..., k] * N(
                batch[i1, i2, ..., :] | mus[i1, i2, ..., k, :], sigmas[i1, i2, ..., k, :]))

        NOTE: The loss is not reduced along the feature dimension (i.e. it should scale ~linearily
        with fs).
        """
        batch = batch.unsqueeze(-2)
        normal_dist = Normal(mus, sigmas)
        g_log_probs = normal_dist.log_prob(batch)
        g_log_probs = logpi + torch.sum(g_log_probs, dim=-1)
        max_log_probs = torch.max(g_log_probs, dim=-1, keepdim=True)[0]
        g_log_probs = g_log_probs - max_log_probs

        g_probs = torch.exp(g_log_probs)
        probs = torch.sum(g_probs, dim=-1)

        log_prob = max_log_probs.squeeze() + torch.log(probs)
        if reduce:
            return - torch.mean(log_prob)
        return - log_prob

    train = partial(data_pass, train=True)
    test = partial(data_pass, train=False)

    for e in range(max_train_epochs):
        cur_best = None
        train(e)
        test_loss = test(e)
        scheduler.step(test_loss)
        earlystopping.step(test_loss)

        is_best = not cur_best or test_loss < cur_best
        if is_best:
            cur_best = test_loss
        checkpoint_fname = os.path.join(rnn_dir, 'checkpoint.tar')
        save_checkpoint({
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'earlystopping': earlystopping.state_dict(),
            "precision": test_loss,
            "epoch": e}, is_best, checkpoint_fname,
            rnn_file)

        if earlystopping.stop:
            print("End of Training because of early stopping at epoch {}".format(e))
            break


def get_mdrnn_cell(rnn_dir):
    rnn_file = os.path.join(rnn_dir, 'best.tar')
    assert os.path.exists(rnn_file)
    state = torch.load(rnn_file)
    mdrnn_cell = MDRNNCell(LSIZE, ASIZE, RSIZE, 5).to(device)
    mdrnn_cell.load_state_dict(
        {k.strip('_l0'): v for k, v in state['state_dict'].items()})
    return mdrnn_cell

def controller_train_proc(ctrl_dir, controller, vae, mdrnn, target_return=950, skip_train=False, display=True):
    # define current best and load parameters
    cur_best = None
    if not os.path.exists(ctrl_dir):
        os.mkdir(ctrl_dir)
    ctrl_file = os.path.join(ctrl_dir, 'best.tar')

    p_queue = Queue()
    r_queue = Queue()
    #e_queue = Queue()   # pipaek : not necessary if not multiprocessing

    print("Attempting to load previous best...")
    if os.path.exists(ctrl_file):
        #state = torch.load(ctrl_file, map_location={'cuda:0': 'cpu'})
        state = torch.load(ctrl_file)
        cur_best = - state['reward']
        controller.load_state_dict(state['state_dict'])
        print("Previous best was {}...".format(-cur_best))

    if skip_train:
        return   # pipaek : 트레이닝을 통한 모델 개선을 skip하고 싶을 때..

    def evaluate(solutions, results, rollouts=10):  # pipaek : rollout 100 -> 10
        """ Give current controller evaluation.

        Evaluation is minus the cumulated reward averaged over rollout runs.

        :args solutions: CMA set of solutions
        :args results: corresponding results
        :args rollouts: number of rollouts

        :returns: minus averaged cumulated reward
        """
        index_min = np.argmin(results)
        best_guess = solutions[index_min]
        restimates = []

        for s_id in range(rollouts):
            print('p_queue.put s_id : %d' % s_id)
            p_queue.put((s_id, best_guess))

        for _ in range(C_POP_SIZE):
            print('>>>rollout_routine!!')
            rollout_routine()  #

        print(">>>Evaluating...")
        for _ in tqdm(range(rollouts)):
            #while r_queue.empty():
            #    sleep(.1)   # pipaek : multi-process가 아니므로
            print('r_queue.get()')
            restimates.append(r_queue.get()[1])

        return best_guess, np.mean(restimates), np.std(restimates)

    def rollout_routine():
        """ Thread routine.

        Threads interact with p_queue, the parameters queue, r_queue, the result
        queue and e_queue the end queue. They pull parameters from p_queue, execute
        the corresponding rollout, then place the result in r_queue.

        Each parameter has its own unique id. Parameters are pulled as tuples
        (s_id, params) and results are pushed as (s_id, result).  The same
        parameter can appear multiple times in p_queue, displaying the same id
        each time.

        As soon as e_queue is non empty, the thread terminate.

        When multiple gpus are involved, the assigned gpu is determined by the
        process index p_index (gpu = p_index % n_gpus).

        :args p_queue: queue containing couples (s_id, parameters) to evaluate
        :args r_queue: where to place results (s_id, results)
        :args e_queue: as soon as not empty, terminate
        :args p_index: the process index
        """
        # init routine
        #gpu = p_index % torch.cuda.device_count()
        #device = torch.device('cuda:{}'.format(gpu) if torch.cuda.is_available() else 'cpu')

        # redirect streams
        #if not os.path.exists(tmp_dir):
        #    os.mkdir(tmp_dir)

        #sys.stdout = open(os.path.join(tmp_dir, 'rollout.out'), 'a')
        #sys.stderr = open(os.path.join(tmp_dir, 'rollout.err'), 'a')

        with torch.no_grad():
            r_gen = RolloutGenerator(vae, mdrnn, controller, device, rollout_time_limit)

            #while e_queue.empty():
            if p_queue.empty():
                return    # pipaek : multi-process가 아니므로, rollout 요청건이 소진되면 그냥 리턴하면 된다.
            else:
                print('p_queue.get()')
                s_id, params = p_queue.get()
                print('r_queue.put() sid=%d' % s_id)
                r_queue.put((s_id, r_gen.rollout(params)))
                print('r_gen.rollout OK')


    parameters = controller.parameters()
    es = cma.CMAEvolutionStrategy(flatten_parameters(parameters), 0.1, {'popsize': C_POP_SIZE})
    print("CMAEvolutionStrategy start OK!!")

    epoch = 0
    log_step = 3
    while not es.stop():
        if cur_best is not None and - cur_best > target_return:
            print("Already better than target, breaking...")
            break

        r_list = [0] * C_POP_SIZE  # result list
        solutions = es.ask()
        print("CMAEvolutionStrategy-ask")

        # push parameters to queue
        for s_id, s in enumerate(solutions):
            #for _ in range(C_N_SAMPLES):
            for _ in range(C_POP_SIZE * C_N_SAMPLES):
                print('in rollout p_queue.put s_id : %d' % s_id)
                p_queue.put((s_id, s))
                #print("p_queue.put %d" % s_id)
                rollout_routine()
                print("rollout_routine OK")

        # retrieve results
        if display:
            pbar = tqdm(total=C_POP_SIZE * C_N_SAMPLES)
        for _ in range(C_POP_SIZE * C_N_SAMPLES):
            #while r_queue.empty():
            #    sleep(.1)
            r_s_id, r = r_queue.get()
            r_list[r_s_id] += r / C_N_SAMPLES
            if display:
                pbar.update(1)
        if display:
            pbar.close()

        es.tell(solutions, r_list)
        es.disp()

        # evaluation and saving
        if epoch % log_step == log_step - 1:
            best_params, best, std_best = evaluate(solutions, r_list, rollouts=10)  # pipaek : evaluate을 위해서 rollout은 10번만 하자..
            print("Current evaluation: {}".format(best))
            if not cur_best or cur_best > best:
                cur_best = best
                print("Saving new best with value {}+-{}...".format(-cur_best, std_best))
                load_parameters(best_params, controller)
                torch.save(
                    {'epoch': epoch,
                     'reward': - cur_best,
                     'state_dict': controller.state_dict()},
                    os.path.join(ctrl_dir, 'best.tar'))
            if - best > target_return:
                print("Terminating controller training with value {}...".format(best))
                break

        epoch += 1

    print("es.stop!!")
    es.result_pretty()
    #e_queue.put('EOP')


def controller_test_proc(controller, vae, mdrnn):
    # define current best and load parameters
    if not os.path.exists(ctrl_dir):
        os.mkdir(ctrl_dir)
    ctrl_file = os.path.join(ctrl_dir, 'best.tar')

    print("Attempting to load previous best...")
    if os.path.exists(ctrl_file):
        # state = torch.load(ctrl_file, map_location={'cuda:0': 'cpu'})
        state = torch.load(ctrl_file)
        controller.load_state_dict(state['state_dict'])

    print("Controller Test Rollout START!!")
    with torch.no_grad():
        r_gen = RolloutGenerator(vae, mdrnn, controller, device, rollout_time_limit)
        r_gen.rollout(flatten_parameters(controller.parameters()), render=True)



# 1. Random Rollout 수행을 통한 experience data 확보
generate_random_rollout_data(random_rollout_dir, random_rollout_num)

# 2-1. VAE를 train할 dataset 생성
v_dataset_train, v_dataset_test = make_vae_dataset(rollout_root_dir)

# 2-2. VAE 모델(V) 생성
v_model = VAE(3, LSIZE).to(device)
v_optimizer = optim.Adam(v_model.parameters())
v_scheduler = ReduceLROnPlateau(v_optimizer, 'min', factor=0.5, patience=10)
v_earlystopping = EarlyStopping('min', patience=5)  # patience 30 -> 10

# 2-3. VAE 모델(V) 훈련
v_model_train_proc(vae_dir, v_model, v_dataset_train, v_dataset_test, v_optimizer, v_scheduler, v_earlystopping, skip_train=False, max_train_epochs=1000)

# 3-1. MDN-RNN를 train할 (random) dataset 생성
m_dataset_train, m_dataset_test = make_mdrnn_dataset(rollout_root_dir)

# 3-2. MDN-RNN 모델(M) 생성
m_model = MDRNN(LSIZE, ASIZE, RSIZE, 5).to(device)    #  pipaek : why gaussian=5?
m_optimizer = torch.optim.RMSprop(m_model.parameters(), lr=1e-3, alpha=.9)
m_scheduler = ReduceLROnPlateau(m_optimizer, 'min', factor=0.5, patience=5)
m_earlystopping = EarlyStopping('min', patience=5)   # patience 30 -> 5

# 3-3. MDN-RNN 모델(M) 훈련
m_model_train_proc(rnn_dir, m_model, v_model, m_dataset_train, m_dataset_test, m_optimizer, m_scheduler, m_earlystopping, skip_train=False, max_train_epochs=20)
m_model_cell = get_mdrnn_cell(rnn_dir).to(device)

# 4-1. Controller 모델(C) 생성
controller = Controller(LSIZE, RSIZE, ASIZE).to(device)

# 4-2. Controller 모델(C) 훈련
controller_train_proc(ctrl_dir, controller, v_model, m_model_cell, skip_train=False)

# 4-3. Controller 모델(C) 시연 (optional)
controller_test_proc(controller, v_model, m_model_cell)

