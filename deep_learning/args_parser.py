import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('data_dir', type=str, default=os.getcwd(), help='Location of directory with data for image classifier to train and test')
parser.add_argument('--arch', action='store', type=str, default='densenet121', nargs='?', help='Choose among 3 pretrained networks')
parser.add_argument('--learning_rate', action='store', type=float, default=0.001, nargs='?', help="learning rate of the optimizer")
parser.add_argument('--hidden_units', action='store', type=int, default=512, nargs='?', help="hidden units")
parser.add_argument('--epochs', action='store', type=int, default=8, nargs='?', help="epochs")
parser.add_argument('--save_dir', action='store', type=str, default='.', nargs='?', help="learning rate of the optimizer")
parser.add_argument('--gpu', action="store_true", default=False, help=" GPU")

args = parser.parse_args()

print("Training will be performed with the settings as followed:")
print("GPU: " + str(args.gpu))
print("epochs: " + str(args.epochs))
print("Learning Rate: " + str(args.learning_rate))
print("Pretrained model: " + args.arch)
print("Hidden units: " + str(args.hidden_units))
print("Saving model to: " + str(args.save_dir))
print("datadir: " + str(args.data_dir))