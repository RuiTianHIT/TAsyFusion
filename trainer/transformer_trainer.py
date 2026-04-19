
import numpy as np
import torch
import torch.nn as nn
from base import BaseTrainer
from torchvision import utils
from model.loss import mse_loss, multi_scale_grad_loss, perceptual_loss_fc, temporal_consistency_loss_fc
from utils.training_utils import select_evenly_spaced_elements, plot_grad_flow, plot_grad_flow_bars
import torch.nn.functional as f
from scipy.ndimage import zoom
import matplotlib.pyplot as plt
from pprint import pprint
metrics_keywords = [
    f"_abs_rel_diff",
    f"_squ_rel_diff",
    f"_RMS_linear",
    f"_RMS_log",
    f"_SILog",
    f"_mean_target_depth",
    f"_median_target_depth",
    f"_mean_prediction_depth",
    f"_median_prediction_depth",
    f"_mean_depth_error",
    f"_median_diff",
    f"_threshold_delta_1.25",
    f"_threshold_delta_1.25^2",
    f"_threshold_delta_1.25^3",
    f"_10_mean_target_depth",
    f"_10_median_target_depth",
    f"_10_mean_prediction_depth",
    f"_10_median_prediction_depth",
    f"_10_abs_rel_diff",
    f"_10_squ_rel_diff",
    f"_10_RMS_linear",
    f"_10_RMS_log",
    f"_10_SILog",
    f"_10_mean_depth_error",
    f"_10_median_diff",
    f"_10_threshold_delta_1.25",
    f"_10_threshold_delta_1.25^2",
    f"_10_threshold_delta_1.25^3",
    f"_20_abs_rel_diff",
    f"_20_squ_rel_diff",
    f"_20_RMS_linear",
    f"_20_RMS_log",
    f"_20_SILog",
    f"_20_mean_target_depth",
    f"_20_median_target_depth",
    f"_20_mean_prediction_depth",
    f"_20_median_prediction_depth",
    f"_20_mean_depth_error",
    f"_20_median_diff",
    f"_20_threshold_delta_1.25",
    f"_20_threshold_delta_1.25^2",
    f"_20_threshold_delta_1.25^3",
    f"_30_abs_rel_diff",
    f"_30_squ_rel_diff",
    f"_30_RMS_linear",
    f"_30_RMS_log",
    f"_30_SILog",
    f"_30_mean_target_depth",
    f"_30_median_target_depth",
    f"_30_mean_prediction_depth",
    f"_30_median_prediction_depth",
    f"_30_mean_depth_error",
    f"_30_median_diff",
    f"_30_threshold_delta_1.25",
    f"_30_threshold_delta_1.25^2",
    f"_30_threshold_delta_1.25^3",
]

def add_to_metrics(idx, metrics, target_, prediction_, mask, event_frame=None, prefix="", debug=False,
                   output_folder=None):
    if len(metrics) == 0:
        metrics = {k: 0 for k in metrics_keywords}

    # prediction_mask = (prediction_ > 0) & (prediction_ < np.amax(target_[~np.isnan(target_)]))
    # depth_mask = np.logical_and(gt > args.min_depth, gt < args.max_depth) # make (target> 3) for mvsec might drives
    # mask = mask & depth_mask
    eps = 1e-5

    target = target_[
        mask]  # np.where(mask, target_, np.max(target_[~np.isnan(target_)]))# target_[mask] but without lossing shape
    prediction = prediction_[
        mask]  # np.where(mask, prediction_, np.max(target_[~np.isnan(target_)]))# prediction_[mask] but without lossing shape

    # thresholds
    # ratio = np.max(np.stack([target/(prediction+eps),prediction/(target+eps)]), axis=0)
    ratio = np.maximum((target / (prediction + eps)), (prediction / (target + eps)))

    new_metrics = {}
    new_metrics[f"{prefix}threshold_delta_1.25"] = np.mean(ratio <= 1.25)
    new_metrics[f"{prefix}threshold_delta_1.25^2"] = np.mean(ratio <= 1.25 ** 2)
    new_metrics[f"{prefix}threshold_delta_1.25^3"] = np.mean(ratio <= 1.25 ** 3)

    # abs diff
    # log_diff = np.log(target+eps)-np.log(prediction+eps)
    log_diff = np.log(prediction + eps) - np.log(target + eps)
    # log_diff = np.abs(log_target - log_prediction)
    abs_diff = np.abs(target - prediction)
    new_metrics[f"{prefix}abs_rel_diff"] = (abs_diff / (target + eps)).mean()
    new_metrics[f"{prefix}squ_rel_diff"] = (abs_diff ** 2 / (target ** 2 + eps)).mean()
    new_metrics[f"{prefix}RMS_linear"] = np.sqrt((abs_diff ** 2).mean())
    new_metrics[f"{prefix}RMS_log"] = np.sqrt((log_diff ** 2).mean())
    new_metrics[f"{prefix}SILog"] = (log_diff ** 2).mean() - (log_diff.mean()) ** 2
    new_metrics[f"{prefix}mean_target_depth"] = target.mean()
    new_metrics[f"{prefix}median_target_depth"] = np.median(target)
    new_metrics[f"{prefix}mean_prediction_depth"] = prediction.mean()
    new_metrics[f"{prefix}median_prediction_depth"] = np.median(prediction)
    new_metrics[f"{prefix}mean_depth_error"] = abs_diff.mean()
    new_metrics[f"{prefix}median_diff"] = np.abs(np.median(target) - np.median(prediction))
    for k, v in new_metrics.items():
        metrics[k] += v
    return metrics
def prepare_depth_data(target, prediction, clip_distance, reg_factor=3.70378):
    # normalize prediction (0 - 1)
    prediction = np.exp(
        reg_factor * (prediction - np.ones((prediction.shape[0], prediction.shape[1], prediction.shape[2]),
                                           dtype=np.float32)))
    target = np.exp(
        reg_factor * (target - np.ones((target.shape[0], target.shape[1], target.shape[2]), dtype=np.float32)))

    # Get back to the absolute values
    target *= clip_distance
    prediction *= clip_distance

    min_depth = np.exp(-1 * reg_factor) * clip_distance
    max_depth = clip_distance

    prediction[np.isinf(prediction)] = max_depth
    prediction[np.isnan(prediction)] = min_depth

    depth_mask = (np.ones_like(target) > 0)
    valid_mask = np.logical_and(target > min_depth, target < max_depth)
    valid_mask = np.logical_and(depth_mask, valid_mask)

    return target, prediction, valid_mask

def quick_norm(img):
    return (img - torch.min(img)) / (torch.max(img) - torch.min(img) + 1e-5)


class TransformerTrainer(BaseTrainer):
    """
    Trainer class

    Note:
        Inherited from BaseTrainer.
        self.optimizer is by default handled by BaseTrainer based on config.
    """

    def __init__(self, model, loss, loss_params, resume, config,
                 data_loader, valid_data_loader=None, train_logger=None):
        super(TransformerTrainer, self).__init__(model, loss,
                                          loss_params, resume, config, train_logger)
        self.config = config
        self.batch_size = data_loader.batch_size
        self.data_loader = data_loader
        self.valid_data_loader = valid_data_loader
        self.valid = True if self.valid_data_loader is not None else False
        #self.log_step = int(np.sqrt(self.batch_size))
        self.log_step = 1
        self.added_tensorboard_graph = False
        self.weight_loss = config['loss']['weight']
        

        if config['use_phased_arch']:
            self.use_phased_arch = True
            print("Using phased architecture")
        else:
            self.use_phased_arch = False
            print("Will not use phased architecture")

        # Parameters for multi scale gradiant loss
        if 'grad_loss' in config:
            self.use_grad_loss = True
            try:
                self.weight_grad_loss = config['grad_loss']['weight']
            except KeyError:
                self.weight_grad_loss = 1.0

            print('Using Multi Scale Gradient loss with weight={:.2f}'.format(
                self.weight_grad_loss))
        else:
            print('Will not use Multi Scale Gradiant loss')
            self.use_grad_loss = False
        
        

    def _to_input_and_target(self, item):
        events = item['event'].float().to(self.gpu)
        target = item['depth'].float().to(self.gpu)
        image = item['image'].float().to(self.gpu)
        flow = item['flow'].float().to(self.gpu)
        semantic = item['semantic'].float().to(self.gpu) if self.use_semantic_loss else None
        times = item['times'].float().to(self.gpu) if self.use_phased_arch else None
        return events, image, flow, target, semantic, times

    @staticmethod
    def make_preview(event_previews, predicted_targets, groundtruth_targets):
        # event_previews: a list of [1 x 1 x H x W] event previews
        # predicted_frames: a list of [1 x 1 x H x W] predicted frames
        # for make_grid, we need to pass [N x 1 x H x W] where N is the number of images in the grid
        return utils.make_grid(torch.cat(event_previews + predicted_targets + groundtruth_targets, dim=0),
                               normalize=False, scale_each=True,
                               nrow=len(predicted_targets))

    @staticmethod
    def make_grad_loss_preview(grad_loss_frames):
        # grad_loss_frames is a list of N multi scale grad losses of size [1 x 1 x H x W]
        return utils.make_grid(torch.cat(grad_loss_frames, dim=0),
                               normalize=True, scale_each=True,
                               nrow=len(grad_loss_frames))

    @staticmethod
    def make_movie(event_previews, predicted_frames, groundtruth_targets):
        # event_previews: a list of [1 x 1 x H x W] event previews
        # predicted_frames: a list of [1 x 1 x H x W] predicted frames
        # for movie, we need to pass [1 x T x 1 x H x W] where T is the time dimension
        video_tensor = None
        for i in torch.arange(len(event_previews)):
            # voxel = quick_norm(event_previews[i])
            voxel = event_previews[i]
            predicted_frame = predicted_frames[i]  # quick_norm(predicted_frames[i])
            movie_frame = torch.cat([voxel,
                                     predicted_frame,
                                     groundtruth_targets[i]],
                                    dim=-1)
            movie_frame.unsqueeze_(dim=0)
            video_tensor = movie_frame if video_tensor is None else \
                torch.cat((video_tensor, movie_frame), dim=1)
        return video_tensor

    def calculate_total_batch_loss(self, loss_dict, total_loss_dict, L):
        nominal_loss = self.weight_loss * sum(loss_dict['losses']) / float(L)
        #print("total si loss of batch: ", nominal_loss)

        losses = []
        losses.append(nominal_loss)

        # Add multi scale gradient loss
        if self.use_grad_loss:
            grad_loss = self.weight_grad_loss * sum(loss_dict['grad_losses']) / float(L)
            losses.append(grad_loss)
            #print("total grad loss of batch: ", grad_loss)

    

        loss = sum(losses)

        # add all losses in a dict for logging
        with torch.no_grad():
            if not total_loss_dict:
                total_loss_dict = {'loss': loss, 'L_si': nominal_loss}
                if self.use_grad_loss:
                    total_loss_dict['L_grad'] = grad_loss
                

            else:
                total_loss_dict['loss'] += loss
                total_loss_dict['L_si'] += nominal_loss
                if self.use_grad_loss:
                    total_loss_dict['L_grad'] += grad_loss
        

        #print("overall total loss: ", total_loss_dict['loss'])
        return total_loss_dict







    def forward_pass_sequence(self, sequence, is_val):
        # 'sequence' is a list containing L successive events <-> depths pairs
        # each element in 'sequence' is a dictionary containing the keys 'events' and 'depth'
        L = len(sequence)
        print("llllllllllllllllllllllllllllllllllllllllllllll", L)
        assert (L > 0)

        total_batch_losses = {}
        
        loss_dict = {'losses': [], 'grad_losses': []}

        # prev_super_states = [None] * self.batch_size      
        self.model.reset_states()
        index = 0
        metrics = {}
        for i, batch_item in enumerate(sequence):
            
            events = batch_item['event']
            target = batch_item['depth']
            image_color = batch_item['rgb']
            text_feature = batch_item['text']

            print(events.shape)
            events = events.float().to(self.gpu) 
            target = target.float().to(self.gpu)
            image_color = image_color.float().to(self.gpu)
            # events = events.float().cuda(non_blocking=True)
            # target = target.float().cuda(non_blocking=True)
            
            pred_dict = self.model(events, image_color, text_feature)
            pred_depth = pred_dict['pred_depth']

            if is_val:
                target_depth, predicted_depth, valid_mask = prepare_depth_data(target.squeeze().cpu().numpy(),
                                                                                    pred_depth.squeeze().cpu().numpy(),
                                                                                    80.0)
                assert predicted_depth.shape == target_depth.shape
                metrics = add_to_metrics(i, metrics, target_depth, predicted_depth, valid_mask, event_frame=None,
                                         prefix="_",
                                         debug=False, output_folder=None)

                for depth_threshold in [10, 20, 30]:
                    depth_threshold_mask = (np.nan_to_num(target_depth) < depth_threshold)
                    add_to_metrics(-1, metrics, target_depth, predicted_depth, valid_mask & depth_threshold_mask,
                                   prefix=f"_{depth_threshold}_", debug=False)




            #calculate loss
            if self.loss_params is not None:
                loss_dict['losses'].append(
                    self.loss(pred_depth, target, **self.loss_params))
            else:
                loss_dict['losses'].append(self.loss(pred_depth, target))
             # Compute the multi scale gradient loss
            if self.use_grad_loss:                
                grad_loss = multi_scale_grad_loss(pred_depth, target)
                loss_dict['grad_losses'].append(grad_loss)

        if is_val:
            index = index + 1
            pprint({k: v / L for k, v in metrics.items()})
            line1 = str({k: v / L for k, v in metrics.items()})
            with open(f"{index}.txt", "w") as f:
                f.write(line1 + "\n")
            print(f"数据已保存到 {index}.txt")




        
        total_batch_losses = self.calculate_total_batch_loss(loss_dict, total_batch_losses, L)
            #print("total batch loss: ", key, total_batch_losses['loss'])

        if is_val:
            return total_batch_losses, metrics
        else:
            return total_batch_losses


    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Current training epoch.
        :return: A log that contains all information you want to save.

        """

        self.model.train()

        all_losses_in_batch = {}
        is_train = ""
        # for batch_idx in range(10):
        #     sequence = self.data_loader.__getitem__(batch_idx)
        # print(len(self.data_loader))
        for batch_idx, sequence in enumerate(self.data_loader):
            # if batch_idx > 10:
            #     break
            self.optimizer.zero_grad()

            losses = self.forward_pass_sequence(sequence, False)
            loss = losses['loss']
            # print(loss)
            # loss_images.backward(retain_graph=True)
            loss.backward()
            if batch_idx % 25 == 0:
                plot_grad_flow(self.model.named_parameters())
            nn.utils.clip_grad_norm_(self.model.parameters(), 0.1)
            self.optimizer.step()
            self.lr_scheduler.step()
            
            with torch.no_grad():
                for loss_name, loss_value in losses.items():
                    if loss_name not in all_losses_in_batch:
                        all_losses_in_batch[loss_name] = []
                    all_losses_in_batch[loss_name].append(loss_value.item())

                if self.verbosity >= 2 and batch_idx % self.log_step == 0:
                    loss_str = ''
                    for loss_name, loss_value in losses.items():
                        loss_str += '{}: {:.4f} '.format(loss_name, loss_value.item())
                    self.logger.info('Train Epoch: {}, batch_idx: {}, [{}/{} ({:.0f}%)] {}'.format(
                        epoch, batch_idx,
                        batch_idx * self.data_loader.batch_size,
                        len(self.data_loader) * self.data_loader.batch_size,
                        100.0 * batch_idx / len(self.data_loader),
                        loss_str))  
                    # print(loss_str)

        # compute average losses over the batch
        total_losses = {loss_name: sum(loss_values) / len(self.data_loader)
                        for loss_name, loss_values in all_losses_in_batch.items()}
        log = {
            'loss': total_losses['loss'],
            'losses': total_losses
            # 'metrics': (total_metrics / self.num_previews).tolist(),
            # 'previews': previews
        }

        if self.valid:
            val_log = self._valid_epoch(epoch=epoch)
            log = {**log, **val_log}

        return log

    def _valid_epoch(self, epoch=0):
        """
        Validate after training an epoch

        :return: A log that contains information about validation

        Note:
            The validation metrics in log must have the key 'val_metrics'.
        """
        is_train = ""
        mean_val = 0.0
        results_dict =[]
        self.model.eval()
        all_losses_in_batch = {}
        with torch.no_grad():
            for batch_idx, sequence in enumerate(self.valid_data_loader):
                # if batch_idx > 10:
                #     break

                losses, results_set = self.forward_pass_sequence(sequence, True)
                results_dict.append(results_set)
                for loss_name, loss_value in losses.items():
                    if loss_name not in all_losses_in_batch:
                        all_losses_in_batch[loss_name] = []
                    all_losses_in_batch[loss_name].append(loss_value.item())

                if self.verbosity >= 2 and batch_idx % self.log_step == 0:
                    self.logger.info('Validation: [{}/{} ({:.0f}%)]'.format(
                        batch_idx * self.valid_data_loader.batch_size,
                        len(self.valid_data_loader) * self.valid_data_loader.batch_size,
                        100.0 * batch_idx / len(self.valid_data_loader)))
            # print("all losses in batch in validation: ", all_losses_in_batch)


        total_losses = {loss_name: sum(loss_values) / len(self.valid_data_loader)
                        for loss_name, loss_values in all_losses_in_batch.items()}



        total_val = 0
        count = 0
        target_key = '_threshold_delta_1.25'
        for d_index in results_dict:
            # d_index 在这里其实是列表中的每一个字典
            if target_key in d_index:
                total_val += d_index[target_key]
                print("d_index[target_key]", d_index[target_key])
                count += 1
        if count > 0:
            mean_val = total_val / count
            print(f"{target_key} 的均值是: {mean_val}")
        else:
            print("未找到相关数据")
        results_dict.clear()



        return {'val_loss': total_losses['loss'],
                'val_losses': total_losses,
                'd1': mean_val}
