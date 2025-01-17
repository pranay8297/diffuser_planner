import pdb

import diffuser.sampling as sampling
import diffuser.utils as utils


#-----------------------------------------------------------------------------#
#----------------------------------- setup -----------------------------------#
#-----------------------------------------------------------------------------#

class Parser(utils.Parser):
    dataset: str = 'walker2d-medium-replay-v2'
    config: str = 'config.locomotion'

args = Parser().parse_args('plan')

sampling_fn = sampling.n_step_guided_p_sample

if args.method == 'ddim':
    # we need 2 things - one is ddim flag and other is number of sampling steps
    sampling_fn = sampling.n_step_guided_p_sample_ddim
#-----------------------------------------------------------------------------#
#---------------------------------- loading ----------------------------------#
#-----------------------------------------------------------------------------#

## load diffusion model and value function from disk
diffusion_experiment = utils.load_diffusion(
    args.loadbase, args.dataset, args.diffusion_loadpath,
    epoch=args.diffusion_epoch, seed=args.seed,
)

args.value_loadpath = 'values/defaults_H4_T20_d0.99'

value_experiment = utils.load_diffusion(
    args.loadbase, args.dataset, args.value_loadpath,
    epoch=args.value_epoch, seed=args.seed,
)

## ensure that the diffusion model and value function are compatible with each other
utils.check_compatibility(diffusion_experiment, value_experiment)

diffusion = diffusion_experiment.ema
dataset = diffusion_experiment.dataset
renderer = diffusion_experiment.renderer

## initialize value guide
value_function = value_experiment.ema
guide_config = utils.Config(args.guide, model=value_function, verbose=False)
guide = guide_config()

logger_config = utils.Config(
    utils.Logger,
    renderer=renderer,
    logpath=args.savepath,
    vis_freq=args.vis_freq,
    max_render=args.max_render,
)

## policies are wrappers around an unconditional diffusion model and a value guide
policy_config = utils.Config(
    args.policy,
    guide=guide,
    scale=args.scale,
    diffusion_model=diffusion,
    normalizer=dataset.normalizer,
    preprocess_fns=args.preprocess_fns,
    ## sampling kwargs
    sample_fn=sampling_fn,
    n_guide_steps=args.n_guide_steps,
    t_stopgrad=args.t_stopgrad,
    scale_grad_by_std=args.scale_grad_by_std,
    n_sampling_steps = args.n_sampling_steps,
    n_concecutive_actions = args.n_concecutive_actions, 
    verbose=False,
    method=args.method,
)

logger = logger_config()
policy = policy_config()


#-----------------------------------------------------------------------------#
#--------------------------------- main loop ---------------------------------#
#-----------------------------------------------------------------------------#

env = dataset.env
observation = env.reset()

## observations for rendering
rollout = [observation.copy()]
ep_actions = []
ep_states = []

total_reward = 0

step = 0

for t in range(args.max_episode_length):

    # if t % 10 == 0: print(args.savepath, flush=True)

    ## save state for rendering only
    

    ## format current observation for conditioning
    conditions = {0: observation}
    actions, samples = policy(conditions, batch_size=args.batch_size, verbose=args.verbose)

    ## execute action in environment
    # loop here for total number of n_concecutive steps and log instead of calling the function for every timestep

    for it in range(actions.shape[0]): # loop through total number of actions

        state = env.state_vector().copy()
        
        action = actions[it] # get the action at that step from horizon
        
        ep_actions.append(action)

        next_observation, reward, terminal, _ = env.step(action)

        ep_states.append(env.state_vector().copy())

        ## print reward and score
        total_reward += reward
        score = env.get_normalized_score(total_reward)
        print(
            f't: {step} | r: {reward:.2f} |  R: {total_reward:.2f} | score: {score:.4f} | ',
            flush=True,
        )

        step += 1

        ## update rollout observations
        rollout.append(next_observation.copy())

        ## render every `args.vis_freq` steps
        # uncomment this after you figure out the error
        # logger.log(step, samples, state, rollout)

        if terminal:
            break

        observation = next_observation

## write results to json file at `args.savepath`

def create_video_from_states(states, env, save_path = f'./logs/hf_cheetah_epl_{step}_ds_{args.n_sampling_steps}_maxR_{int(total_reward)}.mp4'):

    from PIL import Image 
    import cv2

    frame_size = (500, 500)  # Adjust based on your rendering size
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(save_path, fourcc, 22.0, frame_size)

    qpos_dim = env.sim.data.qpos.size
    
    for state in states:

        env.set_state(state[:qpos_dim], state[qpos_dim:])
        frame = env.render(mode='rgb_array')
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame)
    
    out.release()

create_video_from_states(ep_states, env)

logger.finish(step, score, total_reward, terminal, diffusion_experiment, value_experiment)
