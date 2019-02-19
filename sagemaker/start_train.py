import argparse

import sagemaker
from boto3.session import Session
from sagemaker import LocalSession
from sagemaker.chainer.estimator import Chainer


IMAGE = '398271760466.dkr.ecr.ap-northeast-1.amazonaws.com/roboschool_chainerrl:cpu'
DEFAULT_INSTANCE_TYPE = 'ml.m4.4xlarge'
DEFAULT_REGION = 'ap-northeast-1'
DEFAULT_RUNTIME = 24 * 60 * 60  # 1day


def main(parser=argparse.ArgumentParser()):
    import logging
    logging.basicConfig(level=logging.WARN)

    parser.add_argument('--profile', type=str)
    parser.add_argument('--local-mode', action='store_true')
    parser.add_argument('--instance-type', type=str, default=DEFAULT_INSTANCE_TYPE)
    parser.add_argument('--region', type=str, default=DEFAULT_REGION)
    parser.add_argument('--arn', type=str, default=None)
    parser.add_argument('--max-runtime', type=int, default=DEFAULT_RUNTIME, help='seconds')
    args = parser.parse_args()

    boto_session = Session(profile_name=args.profile, region_name=args.region)
    sagemaker_session = sagemaker.Session(boto_session=boto_session)
    role = args.arn if args.arn is not None else sagemaker.get_execution_role(sagemaker_session)

    hyperparameters = {'gpu': 0 if args.instance_type.startswith('ml.p') else -1,
                       'mjcf': 'env/ant_simple.xml',
                       'action-dim': 8,
                       'obs-dim': 28,
                       'skip-step': 25,
                       'algorithm': 'TRPO',
                       'foot-list': 'right_back_foot left_back_foot front_right_foot front_left_foot'
                       }
    chainer_estimator = Chainer(entry_point='train.py',
                                source_dir='../',
                                role=role,
                                image_name=IMAGE,
                                framework_version='5.0.0',
                                sagemaker_session=LocalSession(boto_session) if args.local_mode else sagemaker_session,
                                train_instance_count=1,
                                train_instance_type='local' if args.local_mode else args.instance_type,
                                hyperparameters=hyperparameters,
                                base_job_name='roboschool-chainerrl',
                                train_max_run=args.max_runtime)
    chainer_estimator.fit(wait=args.local_mode)


if __name__ == '__main__':
    main()
