TODO: NaN is detected when CHAINER_DEBUG=1
======================================================

```
CHAINER_DEBUG=1 python train_ant.py \
  --gpu=-1 \
  --eval-n-runs=1 \
  --steps=10000 \
  --fps-window \
  --use-fps-image \
  --update-interval=4 \
  --eval-interval=1 \
  --eval-n-runs=5 \
  --replay-start-size=50 \
  --minibatch-size=2
```

```
Traceback (most recent call last):
  File "/Users/ikeyasu/work/roboschool_chainerrl/train_ant.py", line 204, in <module>
    main()
  File "/Users/ikeyasu/work/roboschool_chainerrl/train_ant.py", line 200, in main
    max_episode_len=timestep_limit)
  File "/Users/ikeyasu/anaconda/envs/chainerrl/lib/python3.6/site-packages/chainerrl/experiments/train_agent.py", line 161, in train_agent_with_evaluation
    logger=logger)
  File "/Users/ikeyasu/anaconda/envs/chainerrl/lib/python3.6/site-packages/chainerrl/experiments/train_agent.py", line 52, in train_agent
    action = agent.act_and_train(obs, r)
  File "/Users/ikeyasu/anaconda/envs/chainerrl/lib/python3.6/site-packages/chainerrl/agents/ddpg.py", line 328, in act_and_train
    self.replay_updater.update_if_necessary(self.t)
  File "/Users/ikeyasu/anaconda/envs/chainerrl/lib/python3.6/site-packages/chainerrl/replay_buffer.py", line 445, in update_if_necessary
    self.update_func(transitions)
  File "/Users/ikeyasu/anaconda/envs/chainerrl/lib/python3.6/site-packages/chainerrl/agents/ddpg.py", line 260, in update
    self.actor_optimizer.update(lambda: self.compute_actor_loss(batch))
  File "/Users/ikeyasu/anaconda/envs/chainerrl/lib/python3.6/site-packages/chainer/optimizer.py", line 655, in update
    loss.backward(loss_scale=self._loss_scale)
  File "/Users/ikeyasu/anaconda/envs/chainerrl/lib/python3.6/site-packages/chainer/variable.py", line 966, in backward
    self._backward_main(retain_grad, loss_scale)
  File "/Users/ikeyasu/anaconda/envs/chainerrl/lib/python3.6/site-packages/chainer/variable.py", line 1095, in _backward_main
    target_input_indexes, out_grad, in_grad)
  File "/Users/ikeyasu/anaconda/envs/chainerrl/lib/python3.6/site-packages/chainer/function_node.py", line 572, in backward_accumulate
    for gx, g_input in six.moves.zip(gxs, grad_inputs)])
  File "/Users/ikeyasu/anaconda/envs/chainerrl/lib/python3.6/site-packages/chainer/function_node.py", line 572, in <listcomp>
    for gx, g_input in six.moves.zip(gxs, grad_inputs)])
  File "/Users/ikeyasu/anaconda/envs/chainerrl/lib/python3.6/site-packages/chainer/functions/math/basic_math.py", line 213, in add
    return Add().apply((lhs, rhs))[0]
  File "/Users/ikeyasu/anaconda/envs/chainerrl/lib/python3.6/site-packages/chainer/function_node.py", line 284, in apply
    raise RuntimeError(msg)
RuntimeError: NaN is detected on forward computation of _ + _
```