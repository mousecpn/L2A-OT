from torch.utils.data import DataLoader
import torch
from resnet import *
import argparse
from torch import nn
from utils.datasets import *
from torch.nn import functional as F
from utils.logger import Logger
from torch import optim
import os
from model import *
from utils.data_reader import *
from torch.autograd import Variable
from utils.utils import *
from utils.wassersteinLoss import *
from torchvision.utils import save_image


os.environ['CUDA_VISIBLE_DEVICES'] = "3,0"

mnist = 'mnist'
mnist_m = 'mnist_m'
svhn = 'svhn'
synth = 'synth'
usps = 'usps'

vlcs_datasets = ["CALTECH", "LABELME", "PASCAL", "SUN"]
pacs_datasets = ["art_painting", "cartoon", "photo", "sketch"]
office_datasets = ["amazon", "dslr", "webcam"]
digits_datasets = [mnist, mnist, svhn, usps]
available_datasets = office_datasets + pacs_datasets + vlcs_datasets + digits_datasets

def get_args():
    train_arg_parser = argparse.ArgumentParser(description="parser")
    train_arg_parser.add_argument("--seed", type=int, default=1,
                                  help="")
    train_arg_parser.add_argument("--test_every", type=int, default=50,
                                  help="")
    train_arg_parser.add_argument("--batch_size", type=int, default=7,
                                  help="")
    train_arg_parser.add_argument("--num_classes", type=int, default=7,
                                  help="")
    train_arg_parser.add_argument("--step_size", type=int, default=1,
                                  help="")
    train_arg_parser.add_argument("--bn_eval", type=int, default=1,
                                  help="")
    train_arg_parser.add_argument("--loops_train", type=int, default=200000,
                                  help="")
    train_arg_parser.add_argument("--unseen_index", type=int, default=0,
                                  help="")
    train_arg_parser.add_argument("--lr", type=float, default=0.0001,
                                  help='')
    train_arg_parser.add_argument("--weight_decay", type=float, default=0.00005,
                                  help='')
    train_arg_parser.add_argument("--momentum", type=float, default=0.9,
                                  help='')
    train_arg_parser.add_argument("--logs", type=str, default='logs/',
                                  help='')
    train_arg_parser.add_argument("--model_path", type=str, default='checkpoints',
                                  help='')
    train_arg_parser.add_argument("--state_dict", type=str, default='',
                                  help='')
    train_arg_parser.add_argument("--data_root", type=str, default="/home/dailh/L2A-OT/data/Train val splits and h5py files pre-read",
                                  help='')
    train_arg_parser.add_argument("--deterministic", type=bool, default=False,
                                  help='')
    args = train_arg_parser.parse_args()


    return args

class L2A_OT_Trainer(object):
    def __init__(self, args, device):
        self.args = args
        self.device = device
        self.beta1 = 0.5
        self.beta2 = 0.999
        self.lr = 0.0001

        root_folder = args.data_root
        train_data = ['art_painting_train.hdf5',
                      'cartoon_train.hdf5',
                      'photo_train.hdf5',
                      'sketch_train.hdf5']

        val_data = ['art_painting_val.hdf5',
                    'cartoon_val.hdf5',
                    'photo_val.hdf5',
                    'sketch_val.hdf5']

        test_data = ['art_painting_test.hdf5',
                     'cartoon_test.hdf5',
                     'photo_test.hdf5',
                     'sketch_test.hdf5']

        self.train_paths = []
        for data in train_data:
            path = os.path.join(root_folder, data)
            self.train_paths.append(path)

        self.val_paths = []
        for data in val_data:
            path = os.path.join(root_folder, data)
            self.val_paths.append(path)

        unseen_index = args.unseen_index

        self.unseen_data_path = os.path.join(root_folder, test_data[unseen_index])
        self.train_paths.remove(self.train_paths[unseen_index])
        self.val_paths.remove(self.val_paths[unseen_index])

        # dataset init
        self.batImageGenTrainsDg = []
        for train_path in self.train_paths:
            batImageGenTrain = BatchImageGenerator(flags=args, file_path=train_path, stage='train',
                                                   b_unfold_label=False)
            self.batImageGenTrainsDg.append(batImageGenTrain)

        self.batImageGenVals = []
        for val_path in self.val_paths:
            batImageGenVal = BatchImageGenerator(flags=args, file_path=val_path, stage='val',
                                                 b_unfold_label=False)
            self.batImageGenVals.append(batImageGenVal)

        self.batImageGenTest = BatchImageGenerator(flags=args, file_path=self.unseen_data_path, stage='test',
                                                   b_unfold_label=False)

        self.n_classes = args.num_classes
        self.num_domains = len(self.batImageGenTrainsDg)
        self.num_aug_domains = self.num_domains
        self.Loss_cls = nn.CrossEntropyLoss()
        self.ReconstructionLoss = nn.L1Loss()
        self.lambda_domain = 1
        self.lambda_cycle = 2
        self.lambda_CE = 1
        self.ckpt_val = self.args.test_every

        # model init
        self.G = Generator(c_dim = 2 * self.num_domains).to(device)



        # self.D = Discriminator(c_dim = self.n_classes).to(device)
        self.g_optimizer = torch.optim.Adam(self.G.parameters(), self.lr, (self.beta1, self.beta2))
        self.best_accuracy_val = 0

    def D_init(self):
        self.D = resnet18(pretrained=False, num_classes=self.num_domains)
        weight = torch.load("/home/dailh/.cache/torch/checkpoints/resnet18-5c106cde.pth")
        weight['fc.weight'] = self.D.state_dict()['fc.weight']
        weight['fc.bias'] = self.D.state_dict()['fc.bias']
        self.D.load_state_dict(weight)
        # self.D = DomianClassifier(domain=3)
        self.D.to(self.device)
        self.d_optimizer = torch.optim.Adam(self.D.parameters(), self.lr, (self.beta1, self.beta2))

        return

    def C_init(self):
        self.C = resnet18(pretrained=False, num_classes=self.n_classes)
        weight = torch.load("/home/dailh/.cache/torch/checkpoints/resnet18-5c106cde.pth")
        weight['fc.weight'] = self.C.state_dict()['fc.weight']
        weight['fc.bias'] = self.C.state_dict()['fc.bias']
        self.C.load_state_dict(weight)
        self.C.to(self.device)
        self.c_optimizer = torch.optim.Adam(self.C.parameters(), self.lr, (self.beta1, self.beta2))

        return

    def DGC_init(self):
        self.DGC = resnet18(pretrained=False, num_classes=self.n_classes)
        weight = torch.load("/home/dailh/.cache/torch/checkpoints/resnet18-5c106cde.pth")
        weight['fc.weight'] = self.DGC.state_dict()['fc.weight']
        weight['fc.bias'] = self.DGC.state_dict()['fc.bias']
        self.DGC.load_state_dict(weight)
        self.DGC.to(self.device)
        # self.DGC.cuda(1)
        self.dgc_optimizer = torch.optim.Adam(self.DGC.parameters(), self.lr, (self.beta1, self.beta2))

        return


    def gradient_penalty(self, y, x):
        """Compute gradient penalty: (L2_norm(dy/dx) - 1)**2."""
        weight = torch.ones(y.size()).to(self.device)
        dydx = torch.autograd.grad(outputs=y,
                                   inputs=x,
                                   grad_outputs=weight,
                                   retain_graph=True,
                                   create_graph=True,
                                   only_inputs=True)[0]

        dydx = dydx.view(dydx.size(0), -1)
        dydx_l2norm = torch.sqrt(torch.sum(dydx**2, dim=1))
        return torch.mean((dydx_l2norm-1)**2)


    def Loss_distribution(self,x_ori,x_gen):
        _,f_ori = self.D(x_ori,latent_flag = True)
        _,f_gen = self.D(x_gen,latent_flag = True)
        C = cost_matrix(f_ori, f_gen).cuda()
        loss = sink(C)
        return loss

    def C_D_loading(self):
        self.D.load_state_dict(torch.load('checkpoints/D.tar')['state'])
        self.C.load_state_dict(torch.load('checkpoints/C.tar')['state'])
        return

    def trainG(self,T):
        self.G.train()
        self.C.eval()
        self.D.eval()
        self.DGC_init()
        self.DGC.train()
        for t in range(T):
            loss_novel = 0.0
            loss_CE = 0.0
            loss_cycle = 0.0
            loss_diversity = 0.0
            fake = []
            rec = []
            for index in range(len(self.batImageGenTrainsDg)):
                # =================================================================================== #
                #                             1. Preprocess input data                                #
                # =================================================================================== #
                x_real, labels = self.batImageGenTrainsDg[index].get_images_labels_batch()
                x_real, labels = torch.from_numpy(
                    np.array(x_real, dtype=np.float32)), torch.from_numpy(
                    np.array(labels, dtype=np.float32))

                # wrap the inputs and labels in Variable
                x_real, labels = Variable(x_real, requires_grad=False).cuda(), \
                                 Variable(labels, requires_grad=False).long().cuda()
                # x_real = x_real.to(self.device)
                label_org = torch.zeros(x_real.size(0),self.num_domains + self.num_aug_domains).cuda()
                label_org[:,index] = 1.0
                new_idx = self.num_domains + index

                label_trg = torch.zeros(x_real.size(0),self.num_domains + self.num_aug_domains).cuda()
                label_trg[:,new_idx] = 1.0

                if index == 0:
                    x_all = x_real
                    labels_all = labels
                else:
                    x_all = torch.cat((x_all,x_real),dim=0)
                    labels_all = torch.cat((labels_all,labels),dim=0)


                # =================================================================================== #
                #                               2. Train the generator                                #
                # =================================================================================== #
                # Original-to-target domain.
                x_fake = self.G(x_real, label_trg)
                fake.append(x_fake)
                loss_novel += self.Loss_distribution(x_real,x_fake)


                x_rec = self.G(x_fake, label_org)
                rec.append(x_rec)
                loss_cycle += self.ReconstructionLoss(x_rec,x_real)

                out_cls = self.C(x_fake)
                loss_CE += self.Loss_cls(out_cls, labels)

            for i in range(self.num_aug_domains):
                for j in range(i,self.num_aug_domains):
                    loss_diversity += self.Loss_distribution(fake[i], fake[j])
            loss_diversity = loss_diversity / 6.0

            total_loss = self.lambda_CE*loss_CE + self.lambda_cycle*loss_cycle - self.lambda_domain*(loss_diversity + loss_novel)

            total_loss.backward()
            print('{}_total_lossG:{}'.format(t,total_loss.item()))

            self.g_optimizer.step()
            self.g_optimizer.zero_grad()
            x_fake_all = torch.cat(fake, dim=0)
            x_rec_all = torch.cat(rec, dim=0)
            loss_DGC = ( self.Loss_cls(self.DGC(x_all), labels_all) + self.Loss_cls(self.DGC(x_fake_all.detach()), labels_all) ) * 0.5
            loss_DGC.backward()
            print('{}_total_lossC:{}'.format(t, loss_DGC.item()))

            self.dgc_optimizer.step()
            self.dgc_optimizer.zero_grad()


            if t % self.ckpt_val == 0:
                torch.save(self.G.state_dict(),  f="checkpoints/G_iteration_{}.pth".format(t))
                self.test_workflow_C(self.DGC,self.batImageGenVals, self.args, t)
                x_all = self.denormalize(x_all)
                x_fake_all = self.denormalize(x_fake_all)
                x_rec_all = self.denormalize(x_rec_all)
                eva_idx  = np.random.randint(0,x_all.size(0))

                save_image(x_all[eva_idx], 'results/real.jpg')

                save_image(x_fake_all[eva_idx], 'results/fake.jpg')

                save_image(x_rec_all[eva_idx], 'results/rec.jpg')
        return

    def trainC(self,T):
        self.C.train()
        for t in range(T):
            # loss_CE = 0.0
            for index in range(len(self.batImageGenTrainsDg)):
                # =================================================================================== #
                #                             1. Preprocess input data                                #
                # =================================================================================== #
                x_real, labels = self.batImageGenTrainsDg[index].get_images_labels_batch()
                x_real, labels = torch.from_numpy(
                    np.array(x_real, dtype=np.float32)), torch.from_numpy(
                    np.array(labels, dtype=np.float32))
                # wrap the inputs and labels in Variable
                x_real, labels = Variable(x_real, requires_grad=False).cuda(), \
                                 Variable(labels, requires_grad=False).long().cuda()
                if index == 0:
                    x_all = x_real
                    labels_all = labels
                else:
                    x_all = torch.cat((x_all,x_real),dim=0)
                    labels_all = torch.cat((labels_all,labels),dim=0)


                # =================================================================================== #
                #                               2. Train the discriminator                            #
                # =================================================================================== #

            out_cls = self.C(x_all)
            loss_CE = self.Loss_cls(out_cls, labels_all)
            # loss_CE = loss_CE/len(self.batImageGenTrainsDg)

            loss_CE.backward()
            self.c_optimizer.step()
            self.c_optimizer.zero_grad()
            print('{}_total_loss:{}'.format(t, loss_CE.item()))


            if t % self.ckpt_val == 0:
                self.test_workflow_C(self.C, self.batImageGenVals, self.args, t)
                # torch.save(self.D.state_dict(),  f="checkpoints/C_iteration_{}.pth".format(t))

        return

    def trainD(self,T):
        self.D.train()
        for t in range(T):
            # loss_CE = 0.0
            for index in range(len(self.batImageGenTrainsDg)):
                # =================================================================================== #
                #                             1. Preprocess input data                                #
                # =================================================================================== #
                x_real, _ = self.batImageGenTrainsDg[index].get_images_labels_batch()
                x_real = torch.from_numpy(np.array(x_real, dtype=np.float32))
                # wrap the inputs and labels in Variable
                x_real = Variable(x_real, requires_grad=False).cuda()
                domain_labels = torch.tensor(int(index)).repeat(x_real.size(0)).cuda()
                if index == 0:
                    x_all = x_real
                    domain_labels_all = domain_labels
                else:
                    x_all = torch.cat((x_all,x_real),dim=0)
                    domain_labels_all = torch.cat((domain_labels_all,domain_labels),dim=0)


                # =================================================================================== #
                #                               2. Train the discriminator                            #
                # =================================================================================== #

            out_cls = self.D(x_all)
            loss_CE = self.Loss_cls(out_cls, domain_labels_all)
            # loss_CE = loss_CE

            loss_CE.backward()
            self.d_optimizer.step()
            self.d_optimizer.zero_grad()
            print('{}_total_loss:{}'.format(t, loss_CE.item()))


            if t % self.ckpt_val == 0:
                self.test_workflow_D(self.batImageGenVals, self.args, t)
                # torch.save(self.D.state_dict(),  f="checkpoints/D_iteration_{}.pth".format(t))

        return

    def test_workflow_C(self, model, batImageGenVals, flags, ite):

        accuracies = []
        for count, batImageGenVal in enumerate(batImageGenVals):
            accuracy_val = self.test_C(model=model,batImageGenTest=batImageGenVal, flags=flags, ite=ite,
                                     log_dir=flags.logs, log_prefix='val_index_{}'.format(count))

            accuracies.append(accuracy_val)

        mean_acc = np.mean(accuracies)

        if mean_acc > self.best_accuracy_val:
            self.best_accuracy_val = mean_acc

            acc_test = self.test_C(model=model,batImageGenTest=self.batImageGenTest, flags=flags, ite=ite,
                                 log_dir=flags.logs, log_prefix='dg_test')

            f = open(os.path.join(flags.logs, 'Best_val.txt'), mode='a')
            f.write(
                'ite:{}, best val accuracy:{}, test accuracy:{}\n'.format(ite, self.best_accuracy_val,
                                                                          acc_test))
            f.close()

            if not os.path.exists(flags.model_path):
                os.makedirs(flags.model_path)

            outfile = os.path.join(flags.model_path, 'best_model_C.tar')
            torch.save({'ite': ite, 'state': model.state_dict()}, outfile)

    # def bn_process(self, flags):
    #     if flags.bn_eval == 1:
    #         self.D.bn_eval()

    def test_C(self, model,flags, ite, log_prefix, log_dir='logs/', batImageGenTest=None):

        # switch on the network test mode
        model.eval()

        if batImageGenTest is None:
            batImageGenTest = BatchImageGenerator(flags=flags, file_path='', stage='test', b_unfold_label=True)

        images_test = batImageGenTest.images
        labels_test = batImageGenTest.labels

        threshold = 50
        if len(images_test) > threshold:

            n_slices_test = int(len(images_test) / threshold)
            indices_test = []
            for per_slice in range(n_slices_test - 1):
                indices_test.append(int(len(images_test) * (per_slice + 1) / n_slices_test))
            test_image_splits = np.split(images_test, indices_or_sections=indices_test)

            # Verify the splits are correct
            test_image_splits_2_whole = np.concatenate(test_image_splits)
            assert np.all(images_test == test_image_splits_2_whole)

            # split the test data into splits and test them one by one
            test_image_preds = []
            for test_image_split in test_image_splits:
                images_test = Variable(torch.from_numpy(np.array(test_image_split, dtype=np.float32))).cuda()
                predictions = model(images_test)

                predictions = predictions.cpu().data.numpy()
                test_image_preds.append(predictions)

            # concatenate the test predictions first
            predictions = np.concatenate(test_image_preds)
        else:
            images_test = Variable(torch.from_numpy(np.array(images_test, dtype=np.float32))).cuda()
            predictions = model(images_test)

            predictions = predictions.cpu().data.numpy()

        accuracy = compute_accuracy(predictions=predictions, labels=labels_test)
        print('----------accuracy test----------:', accuracy)

        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        f = open(os.path.join(log_dir, '{}.txt'.format(log_prefix)), mode='a')
        f.write('ite:{}, accuracy:{}\n'.format(ite, accuracy))
        f.close()

        # switch on the network train mode
        model.train()
        # self.bn_process(flags)

        return accuracy


    def test_workflow_D(self, batImageGenVals, flags, ite):

        accuracies = []
        for count, batImageGenVal in enumerate(batImageGenVals):
            accuracy_val = self.test_D(batImageGenTest=batImageGenVal, flags=flags, ite=ite,domain = count,
                                     log_dir=flags.logs, log_prefix='val_index_{}'.format(count))

            accuracies.append(accuracy_val)

        mean_acc = np.mean(accuracies)

        if mean_acc > self.best_accuracy_val:
            self.best_accuracy_val = mean_acc

            if not os.path.exists(flags.model_path):
                os.makedirs(flags.model_path)

            outfile = os.path.join(flags.model_path, 'D.tar')
            torch.save({'ite': ite, 'state': self.D.state_dict()}, outfile)

    def test_D(self, flags, ite, log_prefix, domain, log_dir='logs/', batImageGenTest=None):

        # switch on the network test mode
        self.D.eval()

        if batImageGenTest is None:
            batImageGenTest = BatchImageGenerator(flags=flags, file_path='', stage='test', b_unfold_label=True)

        images_test = batImageGenTest.images

        threshold = 50
        if len(images_test) > threshold:

            n_slices_test = int(len(images_test) / threshold)
            indices_test = []
            for per_slice in range(n_slices_test - 1):
                indices_test.append(int(len(images_test) * (per_slice + 1) / n_slices_test))
            test_image_splits = np.split(images_test, indices_or_sections=indices_test)

            # Verify the splits are correct
            test_image_splits_2_whole = np.concatenate(test_image_splits)
            assert np.all(images_test == test_image_splits_2_whole)

            # split the test data into splits and test them one by one
            test_image_preds = []
            for test_image_split in test_image_splits:
                images_test = Variable(torch.from_numpy(np.array(test_image_split, dtype=np.float32))).cuda()
                predictions = self.D(images_test)

                predictions = predictions.cpu().data.numpy()
                test_image_preds.append(predictions)

            # concatenate the test predictions first
            predictions = np.concatenate(test_image_preds)
        else:
            images_test = Variable(torch.from_numpy(np.array(images_test, dtype=np.float32))).cuda()
            predictions = self.D(images_test)

            predictions = predictions.cpu().data.numpy()
        domain_label = np.ones((predictions.shape[0],),dtype='int8')*domain

        accuracy = compute_accuracy(predictions=predictions, labels=domain_label)
        print('----------accuracy test----------:', accuracy)

        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        f = open(os.path.join(log_dir, '{}.txt'.format(log_prefix)), mode='a')
        f.write('ite:{}, accuracy:{}\n'.format(ite, accuracy))
        f.close()

        # switch on the network train mode
        self.D.train()
        # self.bn_process(flags)

        return accuracy

    def G_visualize(self):
        self.G.eval()
        self.G.load_state_dict(torch.load('checkpoints/G_iteration_450.pth'))

        for index, batImageGenVal in enumerate(self.batImageGenVals):
            x_real, cls = batImageGenVal.get_images_labels_batch()
            x_real = torch.from_numpy(np.array(x_real, dtype=np.float32))

            # wrap the inputs and labels in Variable
            x_real = Variable(x_real, requires_grad=False).cuda()
            x_real = x_real.to(self.device)
            label_org = torch.zeros(x_real.size(0), self.num_domains + self.num_aug_domains).cuda()
            label_org[:, index] = 1.0
            new_idx = self.num_domains + index

            label_trg = torch.zeros(x_real.size(0), self.num_domains + self.num_aug_domains).cuda()
            label_trg[:, new_idx] = 1.0
            x_fake = self.G(x_real, label_trg)
            x_rec = self.G(x_fake, label_org)
            x_real = self.denormalize(x_real)
            x_fake = self.denormalize(x_fake)
            x_rec = self.denormalize(x_rec)
            save_image(x_real[0],'results/real.jpg')
            save_image(x_fake[0],'results/fake.jpg')
            save_image(x_rec[0], 'results/rec.jpg')

        return

    def denormalize(self,x):
        # x is a tensor
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]
        mean = torch.tensor(mean).cuda()
        std = torch.tensor(std).cuda()

        x *= std.view(1,3,1,1)
        x += mean.view(1, 3, 1, 1)
        return x

def main():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainer = L2A_OT_Trainer(args, device)
    trainer.C_init()
    # trainer.trainC(500)
    trainer.D_init()
    # trainer.trainD(200)
    trainer.C_D_loading()
    trainer.trainG(1000)
    # trainer.G_visualize()

if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    main()