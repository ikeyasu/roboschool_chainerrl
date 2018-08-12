TODO: Invalid operation is performed in: Reshape (Forward)
==========================================================

```
Traceback (most recent call last):
  File "train_ant.py", line 204, in <module>
    main()
  File "train_ant.py", line 200, in main
    max_episode_len=timestep_limit)
  File "/usr/local/lib/python3.5/dist-packages/chainerrl/experiments/train_agent.py", line 147, in train_agent_with_evaluation
    logger=logger)
  File "/usr/local/lib/python3.5/dist-packages/chainerrl/experiments/train_agent.py", line 53, in train_agent
    action = agent.act_and_train(obs, r)
  File "/usr/local/lib/python3.5/dist-packages/chainerrl/agents/ddpg.py", line 328, in act_and_train
    self.replay_updater.update_if_necessary(self.t)
  File "/usr/local/lib/python3.5/dist-packages/chainerrl/replay_buffer.py", line 424, in update_if_necessary
    self.update_func(transitions)
  File "/usr/local/lib/python3.5/dist-packages/chainerrl/agents/ddpg.py", line 259, in update
    self.critic_optimizer.update(lambda: self.compute_critic_loss(batch))
  File "/usr/local/lib/python3.5/dist-packages/chainer/optimizer.py", line 536, in update
    loss = lossfun(*args, **kwds)
  File "/usr/local/lib/python3.5/dist-packages/chainerrl/agents/ddpg.py", line 259, in <lambda>
    self.critic_optimizer.update(lambda: self.compute_critic_loss(batch))
  File "/usr/local/lib/python3.5/dist-packages/chainerrl/agents/ddpg.py", line 200, in compute_critic_loss
    (1.0 - batch_terminal) * F.reshape(next_q, (batchsize,))
  File "/usr/local/lib/python3.5/dist-packages/chainer/functions/array/reshape.py", line 98, in reshape
    y, = Reshape(shape).apply((x,))
  File "/usr/local/lib/python3.5/dist-packages/chainer/function_node.py", line 230, in apply
    self._check_data_type_forward(in_data)
  File "/usr/local/lib/python3.5/dist-packages/chainer/function_node.py", line 298, in _check_data_type_forward
    self.check_type_forward(in_type)
  File "/usr/local/lib/python3.5/dist-packages/chainer/functions/array/reshape.py", line 31, in check_type_forward
    type_check.prod(x_type.shape) == type_check.prod(self.shape))
  File "/usr/local/lib/python3.5/dist-packages/chainer/utils/type_check.py", line 524, in expect
    expr.expect()
  File "/usr/local/lib/python3.5/dist-packages/chainer/utils/type_check.py", line 482, in expect
    '{0} {1} {2}'.format(left, self.inv, right))
chainer.utils.type_check.InvalidType:
Invalid operation is performed in: Reshape (Forward)

Expect: prod(in_types[0].shape) == prod((200,))
Actual: 1 != 200
```

NOTE:

* 200 is default value of minibatch-size
* To reproduce shortly,

```
$ python train_ant.py --gpu=-1 --eval-n-runs=1 --steps=10000 --fps-window --use-fps-image \
 --update-interval=4 --eval-interval=1 --eval-n-runs=5 --replay-start-size=50 --minibatch-size=2
```