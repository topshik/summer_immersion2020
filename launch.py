import argparse

import train


# def load_yaml(filepath: str) -> dict:
#     """
#     Load yaml config
#     :param filepath: Path to config
#     :return: parsed yaml
#     """
#     with open(filepath, "r") as stream:
#         result = yaml.safe_load(stream)
#     return result


# Main parser
parser = argparse.ArgumentParser(description='')
subparsers = parser.add_subparsers(title='commands', help='', required=True, dest='command')

# Training parser
parser_train = subparsers.add_parser('train', description='Trains the model.')
parser_train.add_argument('-o', '--output', type=str, default='checkpoints', help='Directory for model checkpoints.')
parser_train.add_argument('-p', '--prior', type=str, default='SimpleGaussian', help='VAE prior distribution type.')
parser_train.add_argument('-c', '--components', type=int, default=300, help='Number of prior components.')
parser_train.add_argument('-a', '--anneal', type=str, default='logistic', help='Annealing function type: logistic,' 
                                                                               'linear or const.')
parser_train.add_argument('-b', '--beta', type=float, help='KL loss weight, used to anneal to it.')

# Evaluation parser
parser_eval = subparsers.add_parser('validate', description='Evaluates provided model on validation set.')
parser_eval.add_argument('-m', '--model', type=str, required=True, help='path to saved model.')
parser_eval.add_argument('-p', '--prior', type=str, default='SimpleGaussian', help='VAE prior distribution type.')
parser_eval.add_argument('-c', '--components', type=int, default=300, help='number of prior components.')


if __name__ == '__main__':
    args = parser.parse_args()
    # params = load_yaml(args.config) if args.config else {}

    if args.command == 'train':
        train.train(args.prior, args.components, args.anneal, args.beta)

    elif args.command == 'validate':
        train.validate(args.model, args.prior, args.components)
