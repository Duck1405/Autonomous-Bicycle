import pickle
import random
import logging

import cv2
import torch
import numpy as np
from tqdm import tqdm, trange
import time
from lib.video import VideoInference


class Runner:
    def __init__(self, cfg, exp, device, resume=False, view=None, deterministic=False):
        self.cfg = cfg
        self.exp = exp
        self.device = device
        self.resume = resume
        self.view = view
        self.logger = logging.getLogger(__name__)
        # DataLoader workers; configurable so small CPU/debug runs can use 0 (avoids
        # multiprocessing overhead) while GPU servers keep the default of 8.
        self.num_workers = cfg['num_workers'] if 'num_workers' in cfg else 8

        # Fix seeds
        torch.manual_seed(cfg['seed'])
        np.random.seed(cfg['seed'])
        random.seed(cfg['seed'])

        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    def train(self):
        self.exp.train_start_callback(self.cfg)
        starting_epoch = 1
        model = self.cfg.get_model()
        model = model.to(self.device)
        optimizer = self.cfg.get_optimizer(model.parameters())
        scheduler = self.cfg.get_lr_scheduler(optimizer)
        if self.resume:
            last_epoch, model, optimizer, scheduler = self.exp.load_last_train_state(model, optimizer, scheduler)
            starting_epoch = last_epoch + 1
        max_epochs = self.cfg['epochs']
        train_loader = self.get_train_dataloader()
        loss_parameters = self.cfg.get_loss_parameters()
        for epoch in trange(starting_epoch, max_epochs + 1, initial=starting_epoch - 1, total=max_epochs):
            self.exp.epoch_start_callback(epoch, max_epochs)
            model.train()
            pbar = tqdm(train_loader)
            epoch_start = time.time()
            running, n_iters = {}, 0
            for i, (images, labels, _) in enumerate(pbar):
                images = images.to(self.device)
                labels = labels.to(self.device)

                # Forward pass
                outputs = model(images, **self.cfg.get_train_parameters())
                loss, loss_dict_i = model.loss(outputs, labels, **loss_parameters)

                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Scheduler step (iteration based)
                scheduler.step()
                
                for k, v in loss_dict_i.items():
                    running[k] = running.get(k, 0.0) + float(v)
                n_iters += 1
                

                # Log
                postfix_dict = {key: float(value) for key, value in loss_dict_i.items()}
                postfix_dict['lr'] = optimizer.param_groups[0]["lr"]
                self.exp.iter_end_callback(epoch, max_epochs, i, len(train_loader), loss.item(), postfix_dict)
                postfix_dict['loss'] = loss.item()
                pbar.set_postfix(ordered_dict=postfix_dict)
                
                
            epoch_time = time.time() - epoch_start
            means = {k: running[k] / max(n_iters, 1) for k in running}
            it_per_s = n_iters / epoch_time if epoch_time > 0 else 0.0
            eta_min = (max_epochs - epoch) * epoch_time / 60
            summary = ' | '.join('{}: {:.4f}'.format(k, means[k]) for k in means)
            self.logger.info('Epoch [%d/%d] %.1fs (%.2f it/s) | %s | lr: %.2e | ETA: %.1f min',
                            epoch, max_epochs, epoch_time, it_per_s, summary,
                            optimizer.param_groups[0]['lr'], eta_min)
            
            self.exp.epoch_end_callback(epoch, max_epochs, model, optimizer, scheduler)

            # Validate
            if (epoch + 1) % self.cfg['val_every'] == 0:
                self.eval(epoch, on_val=True)
        self.exp.train_end_callback()

    def eval(self, epoch, on_val=False, save_predictions=False):
        model = self.cfg.get_model()
        model_path = self.exp.get_checkpoint_path(epoch)
        self.logger.info('Loading model %s', model_path)
        
        state_dict = self.exp.get_epoch_model(epoch)
        model.load_state_dict(state_dict)
        model = model.to(self.device)
        
        
        model.eval()
        if on_val:
            print("On Validation")
            dataloader = self.get_val_dataloader()
        else:
            print("On Test loader")
            dataloader = self.get_test_dataloader()
            
            
        
        test_parameters = self.cfg.get_test_parameters()
        predictions = []
        self.exp.eval_start_callback(self.cfg)
        with torch.no_grad():
            i = 0
            for idx, (images, _, _) in enumerate(tqdm(dataloader)):
                images = images.to(self.device)
                output = model(images, **test_parameters)
                prediction = model.decode(output, as_lanes=True)
                predictions.extend(prediction)

                if self.view:
                    img = (images[0].cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                    img, fp, fn = dataloader.dataset.draw_annotation(idx, img=img, pred=prediction[0])
                    if self.view == 'mistakes' and fp == 0 and fn == 0:
                        continue
                    cv2.imshow('pred', img)
                    cv2.waitKey(0)
                    print(f"i: {i}")
                    if (i > 5):
                        cv2.destroyAllWindows()
                        break
                    i += 1

        if save_predictions:
            with open('predictions.pkl', 'wb') as handle:
                pickle.dump(predictions, handle, protocol=pickle.HIGHEST_PROTOCOL)
        self.exp.eval_end_callback(dataloader.dataset.dataset, predictions, epoch)
        
        
        
    def get_model(self,epoch):
        model = self.cfg.get_model()
        print("obtained Model:", model)
        model_path = self.exp.get_checkpoint_path(epoch)
        print("obtained Model path", model_path)
        self.logger.info('Loading model %s', model_path)
        
        # Load model on CPU:
        state_dict = self.exp.get_epoch_model(epoch)
        model.load_state_dict(state_dict)
        # model.load_state_dict(self.exp.get_epoch_model(epoch), map_location=torch.device('cpu'))
        model = model.to(self.device)
        model.eval()
        # if on_val:
        #     print("On Validation")
        #     dataloader = self.get_val_dataloader()
        # else:
        #     print("On Test loader")
        #     dataloader = self.get_test_dataloader()
        return model

    def get_train_dataloader(self):
        train_dataset = self.cfg.get_dataset('train')
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                   batch_size=self.cfg['batch_size'],
                                                   shuffle=True,
                                                   num_workers=self.num_workers,
                                                   worker_init_fn=self._worker_init_fn_)
        return train_loader

    def get_test_dataloader(self):
        test_dataset = self.cfg.get_dataset('test')
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                                  batch_size=self.cfg['batch_size'] if not self.view else 1,
                                                  shuffle=False,
                                                  num_workers=self.num_workers,
                                                  worker_init_fn=self._worker_init_fn_)
        return test_loader

    def get_val_dataloader(self):
        val_dataset = self.cfg.get_dataset('val')
        val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                                 batch_size=self.cfg['batch_size'],
                                                 shuffle=False,
                                                 num_workers=self.num_workers,
                                                 worker_init_fn=self._worker_init_fn_)
        return val_loader
    
    def get_video_inference(self,conf_threshold,nms_thres, nms_topk, path_video, output_folder):
        num = self.exp.get_last_checkpoint_epoch()
        wieghts = self.get_model(num)
        video = VideoInference(model_wieghts=wieghts, frame_limit = 99999, video_path = str(path_video / "1.mp4"), view = True, output_folder = output_folder, device = self.device, conf_threshold = conf_threshold, nms_thres = nms_thres, nms_topk = nms_topk)
        video.video_eval()
        video.set_video_path(str(path_video / "2.mp4"))
        video.video_eval()
        video.set_video_path(str(path_video / "3.mp4"))
        video.video_eval()
        

    @staticmethod
    def _worker_init_fn_(_):
        torch_seed = torch.initial_seed()
        np_seed = torch_seed // 2**32 - 1
        random.seed(torch_seed)
        np.random.seed(np_seed)