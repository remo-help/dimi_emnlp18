import configparser
import argparse
import utils
def start_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Imports a saved model and translates it into a readable PCFG')
    parser.add_argument("--path",
                        help="The path to your datafile.",
                        required=True)
    parser.add_argument("--outputfile",
                        help="The name of your outputfile. Default: config.ini",
                        required=False, default='config.ini')
    parser.add_argument("--outputdir",
                        help="The directory for your outputs",
                        required=False, default='./outputs/test')
    parser.add_argument("--iters",
                        help="Number of iters, default = 100",
                        required=False, default=100, type=int)
    parser.add_argument("--k",
                        help="Number of non terminals, default = 6",
                        required=False, default=6, type=int)
    parser.add_argument("--cpu",
                        help="Number of cpu workers, default is max available cpus",
                        required=False, default=0, type=int)
    parser.add_argument("--d",
                        help="Number of center embeddings allowed, default = 2",
                        required=False, default=2, type=int)
    parser.add_argument("--alpha",
                        help="number of symmetric Dirichlet prior, default = 0.2 ",
                        required=False, default=0.2, type=float)
    parser.add_argument("--batch_size",
                        help="Number of batch size per worker",
                        required=False, default=64, type=int)
    return parser


def main():
    parser = start_parser()
    args = parser.parse_args()
    if args.cpu == 0:
        import multiprocessing
        cpu_workers = multiprocessing.cpu_count()
    else:
        cpu_workers = args.cpu
    config = configparser.ConfigParser()
    utils.make_ints_file.make_ints(args.path)
    config['io'] = {'input_file': args.path+'.ints',
                         'output_dir': args.outputdir,
                         'dict_file': args.path+'.dict'}

    config['params'] = {'iters': args.iters,
                         'k': args.k,
                         'init_alpha': args.alpha,
                        'cpu_workers': cpu_workers,
                        'd': args.d,
                        'batch_per_worker': args.batch_size
                        }
    with open(args.outputfile, 'w+') as configfile:
        config.write(configfile)

if __name__ == "__main__":
    main()