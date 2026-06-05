
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F


from tqdm import tqdm
from utilities.utilities import calculate_psnr, calculate_ssim, tensor2img


# modify this template to derive your train class

class Trainer(object):
    def __init__(self, config, reporter):

        self.config     = config
        # logger
        self.reporter   = reporter
        # Data loader
        #============build train dataloader==============#
        # TODO to modify the key: "your_train_dataset" to get your train dataset path
        train_dataset   = config["dataset_paths"][config["dataset_name"]]
        #================================================#
        print("Prepare the train dataloader...")
        dlModulename    = config["dataloader"]
        package         = __import__("data_tools.dataloader_%s"%dlModulename, fromlist=[""])
        dataloaderClass = getattr(package, 'GetLoader')
        
        dataloader      = dataloaderClass(train_dataset,
                                        config["batch_size"],
                                        config["random_seed"],
                                        **config["dataset_params"]
                                    )
        
        self.train_loader= dataloader
 
        #========build evaluation dataloader=============#
        # TODO to modify the key: "your_eval_dataset" to get your evaluation dataset path
        # eval_dataset = config["test_dataset_paths"][config["eval_dataset_name"].lower()]

        #================================================#
        print("Prepare the evaluation dataloader...")
        dlModulename    = config["eval_dataloader"]
        package         = __import__("data_tools.eval_dataloader_%s"%dlModulename, fromlist=[""])
        dataloaderClass = getattr(package, 'EvalDataset')

        #================urban100======================#
        self.eval_loader1      = dataloaderClass('urban100',
                                        config["test_dataset_paths"]['urban100'],
                                        config["eval_batch_size"],
                                        image_scale = config["dataset_params"]["image_scale"]
                                        )
        self.eval_iter1  = len(self.eval_loader1)//config["eval_batch_size"]
        if len(self.eval_loader1)%config["eval_batch_size"]>0:
            self.eval_iter1+=1
        
        #================b100======================#
        self.eval_loader2      = dataloaderClass('b100',
                                        config["test_dataset_paths"]['b100'],
                                        config["eval_batch_size"],
                                        image_scale = config["dataset_params"]["image_scale"]
                                        )
        self.eval_iter2  = len(self.eval_loader2)//config["eval_batch_size"]
        if len(self.eval_loader2)%config["eval_batch_size"]>0:
            self.eval_iter2+=1

        #================set14======================#
        self.eval_loader3      = dataloaderClass('set14',
                                        config["test_dataset_paths"]['set14'],
                                        config["eval_batch_size"],
                                        image_scale = config["dataset_params"]["image_scale"]
                                        )
        self.eval_iter3  = len(self.eval_loader3)//config["eval_batch_size"]
        if len(self.eval_loader3)%config["eval_batch_size"]>0:
            self.eval_iter3+=1


        #================set5======================#
        self.eval_loader4      = dataloaderClass('set5',
                                        config["test_dataset_paths"]['set5'],
                                        config["eval_batch_size"],
                                        image_scale = config["dataset_params"]["image_scale"]
                                        )
        self.eval_iter4  = len(self.eval_loader4)//config["eval_batch_size"]
        if len(self.eval_loader4)%config["eval_batch_size"]>0:
            self.eval_iter4+=1

        #================manga109======================#
        self.eval_loader5      = dataloaderClass('manga109',
                                        config["test_dataset_paths"]['manga109'],
                                        config["eval_batch_size"],
                                        image_scale = config["dataset_params"]["image_scale"]
                                        )
        self.eval_iter5  = len(self.eval_loader5)//config["eval_batch_size"]
        if len(self.eval_loader5)%config["eval_batch_size"]>0:
            self.eval_iter5+=1


        #==============build tensorboard=================#
        if self.config["use_tensorboard"]:
            from utilities.utilities import build_tensorboard
            self.tensorboard_writer = build_tensorboard(self.config["project_summary"])


    # TODO modify this function to build your models
    def __init_framework__(self):
        '''
            This function is designed to define the framework,
            and print the framework information into the log file
        '''
        #===============build models================#
        print("build models...")
        script_name     = "components."+self.config["module_script_name"]
        class_name      = self.config["class_name"]
        package         = __import__(script_name, fromlist=[""])
        network_class   = getattr(package, class_name)

        self.reporter.writeInfo("Model structure:")

        self.network = network_class(3,
                                    3,
                                    self.config["feature_num"],
                                    **self.config["module_params"]
                                    )
        self.reporter.writeModel(self.network.__str__())
        
        # 改进的GPU训练设置
        use_cpu = self.config.get("use_cpu", False)
        if not use_cpu:
            gpu_count = torch.cuda.device_count()
            
            # 将模型移至 GPU
            self.network = self.network.cuda()
            
            # 强制确保所有参数和缓冲区都在GPU上
            print("强制移动所有参数到GPU...")
            
            # 方法1: 使用parameters()和buffers()
            for param in self.network.parameters():
                param.data = param.data.cuda()
            for buf in self.network.buffers():
                buf.data = buf.data.cuda()
            
            # 方法2: 使用named_parameters()和named_buffers()进行深度检查
            print("深度检查所有参数...")
            for name, param in self.network.named_parameters():
                if not param.is_cuda:
                    print(f"发现参数 {name} 在CPU上，强制移动到GPU")
                    param.data = param.data.cuda()
            
            for name, buf in self.network.named_buffers():
                if not buf.is_cuda:
                    print(f"发现缓冲区 {name} 在CPU上，强制移动到GPU")
                    buf.data = buf.data.cuda()
            
            # 方法3: 递归检查所有子模块
            def force_move_to_gpu(module):
                for child in module.children():
                    force_move_to_gpu(child)
                for param in module.parameters(recurse=False):
                    param.data = param.data.cuda()
                for buf in module.buffers(recurse=False):
                    buf.data = buf.data.cuda()
            
            force_move_to_gpu(self.network)
            
            # 方法4: 强制重新注册所有参数到GPU
            print("强制重新注册参数到GPU...")
            for name, module in self.network.named_modules():
                if hasattr(module, '_parameters'):
                    for param_name, param in module._parameters.items():
                        if param is not None and not param.is_cuda:
                            print(f"强制移动模块 {name}.{param_name} 到GPU")
                            param.data = param.data.cuda()
                if hasattr(module, '_buffers'):
                    for buffer_name, buffer in module._buffers.items():
                        if buffer is not None and not buffer.is_cuda:
                            print(f"强制移动模块 {name}.{buffer_name} 到GPU")
                            buffer.data = buffer.data.cuda()
            
            print("参数移动完成")
            
            # 多GPU训练设置
            if gpu_count > 1:
                if self.config.get("distributed", False):
                    # 使用分布式训练
                    print(f"Using DistributedDataParallel with {gpu_count} GPUs")
                    import torch.distributed as dist
                    from torch.nn.parallel import DistributedDataParallel as DDP
                    
                    # 初始化分布式训练
                    dist.init_process_group(backend=self.config.get("dist_backend", "nccl"))
                    torch.cuda.set_device(self.config.get("local_rank", 0))
                    
                    self.network = DDP(self.network, 
                                     device_ids=[self.config.get("local_rank", 0)],
                                     output_device=self.config.get("local_rank", 0))
                else:
                    # 使用DataParallel
                    print(f"Using DataParallel with {gpu_count} GPUs")
                    
                    # 确保模型在包装DataParallel之前完全在GPU上
                    print("DataParallel包装前深度检查...")
                    for name, param in self.network.named_parameters():
                        if not param.is_cuda:
                            print(f"DataParallel前发现参数 {name} 在CPU上，强制移动到GPU")
                            param.data = param.data.cuda()
                    for name, buf in self.network.named_buffers():
                        if not buf.is_cuda:
                            print(f"DataParallel前发现缓冲区 {name} 在CPU上，强制移动到GPU")
                            buf.data = buf.data.cuda()
                    
                    # 强制重新注册所有参数到GPU
                    for name, module in self.network.named_modules():
                        if hasattr(module, '_parameters'):
                            for param_name, param in module._parameters.items():
                                if param is not None and not param.is_cuda:
                                    print(f"DataParallel前强制移动模块 {name}.{param_name} 到GPU")
                                    param.data = param.data.cuda()
                        if hasattr(module, '_buffers'):
                            for buffer_name, buffer in module._buffers.items():
                                if buffer is not None and not buffer.is_cuda:
                                    print(f"DataParallel前强制移动模块 {name}.{buffer_name} 到GPU")
                                    buffer.data = buffer.data.cuda()
                    
                    # 现在包装DataParallel
                    self.network = nn.DataParallel(self.network)
                    
                    # 包装后再次确保所有参数都在GPU上
                    print("DataParallel包装后深度检查...")
                    for name, param in self.network.named_parameters():
                        if not param.is_cuda:
                            print(f"DataParallel后发现参数 {name} 在CPU上，强制移动到GPU")
                            param.data = param.data.cuda()
                    for name, buf in self.network.named_buffers():
                        if not buf.is_cuda:
                            print(f"DataParallel后发现缓冲区 {name} 在CPU上，强制移动到GPU")
                            buf.data = buf.data.cuda()
                    
                    # 包装后强制重新注册所有参数到GPU
                    for name, module in self.network.named_modules():
                        if hasattr(module, '_parameters'):
                            for param_name, param in module._parameters.items():
                                if param is not None and not param.is_cuda:
                                    print(f"DataParallel后强制移动模块 {name}.{param_name} 到GPU")
                                    param.data = param.data.cuda()
                        if hasattr(module, '_buffers'):
                            for buffer_name, buffer in module._buffers.items():
                                if buffer is not None and not buffer.is_cuda:
                                    print(f"DataParallel后强制移动模块 {name}.{buffer_name} 到GPU")
                                    buffer.data = buffer.data.cuda()
            else:
                print("Single GPU training")
                
            print(f"Model moved to GPU. {gpu_count} GPU(s) available.")
        else:
            print("Training on CPU.")
        
        # 加载checkpoint（resume训练）
        if self.config.get("resume", False) and self.config.get("resume_epoch", 0) > 0:
            resume_epoch = self.config["resume_epoch"]
            model_path = os.path.join(self.config["project_checkpoints"],
                                    f"epoch{resume_epoch}_{self.config['checkpoint_names']['generator_name']}.pth")
            if os.path.exists(model_path):
                print(f"加载checkpoint: {model_path}")
                map_location = 'cpu' if use_cpu else 'cuda'
                
                # 加载模型权重
                checkpoint = torch.load(model_path, map_location=map_location)
                
                # 处理DataParallel模型的权重加载
                if isinstance(self.network, nn.DataParallel):
                    # 如果当前模型是DataParallel，但保存的权重不是
                    if not any(key.startswith('module.') for key in checkpoint.keys()):
                        # 添加module前缀
                        new_checkpoint = {}
                        for key, value in checkpoint.items():
                            new_checkpoint[f'module.{key}'] = value
                        checkpoint = new_checkpoint
                else:
                    # 如果当前模型不是DataParallel，但保存的权重是
                    if any(key.startswith('module.') for key in checkpoint.keys()):
                        # 移除module前缀
                        new_checkpoint = {}
                        for key, value in checkpoint.items():
                            if key.startswith('module.'):
                                new_checkpoint[key[7:]] = value
                            else:
                                new_checkpoint[key] = value
                        checkpoint = new_checkpoint
                
                self.network.load_state_dict(checkpoint, strict=False)
                print(f"成功加载epoch {resume_epoch}的模型参数")
            else:
                print(f"警告: checkpoint文件不存在: {model_path}")

        # if in finetune phase, load the pretrained checkpoint
        if self.config["phase"] == "finetune":
            model_path = os.path.join(self.config["imagenet_checkpoints"],
                                        f"epoch{self.config['ckpt']}_{self.config['checkpoint_names']['generator_name']}.pth")
            map_location = 'cpu' if use_cpu else 'cuda'
            self.network.load_state_dict(torch.load(model_path, map_location=map_location), strict=False)
            print('loaded trained backbone model epoch {}...!'.format(self.config["imagenet_checkpoints"]))
        
        # 初始化tensorboard writer
        if self.config["use_tensorboard"]:
            from torch.utils.tensorboard import SummaryWriter
            self.tensorboard_writer = SummaryWriter(log_dir=os.path.join(self.config["project_root"], "tensorboard"))
            print("Tensorboard writer initialized")


    # TODO modify this function to evaluate your model
    def __evaluation__(self, eval_loader, eval_iter, epoch, step = 0):
        # Evaluate the checkpoint
        self.network.eval()
        total_psnr = 0
        total_ssim = 0
        total_num  = 0
        dataset_name = eval_loader.dataset_name
        
        # 模型已经在初始化时移动到GPU，无需重复移动
        
        with torch.no_grad():
            for _ in tqdm(range(eval_iter), desc=f"评估{dataset_name}"):
                hr, lr = eval_loader()
                if not self.config.get("use_cpu", False):
                    hr = hr.cuda()
                    lr = lr.cuda()
                res     = self.network(lr)
                res     = tensor2img(res.cpu())
                hr      = tensor2img(hr.cpu())
                psnr    = calculate_psnr(res[0],hr[0])
                ssim    = calculate_ssim(res[0],hr[0])
                total_psnr+= psnr
                total_ssim+= ssim
                total_num+=1
        final_psnr = total_psnr/total_num
        final_ssim = total_ssim/total_num
        
        # 添加更详细的日志输出
        print(f"📊 [{dataset_name}] PSNR: {final_psnr:.4f}, SSIM: {final_ssim:.5f}")
        
        self.reporter.writeTrainLog(epoch, step, 
            f"Dataset [{dataset_name}] PSNR: {final_psnr:.4f}, SSIM: {final_ssim:.5f}")
        
        if self.config["use_tensorboard"]:
            self.tensorboard_writer.add_scalar(f'metric/{dataset_name}_PSNR', final_psnr, epoch)
            self.tensorboard_writer.add_scalar(f'metric/{dataset_name}_SSIM', final_ssim, epoch)
        
        # 返回评估结果
        return {
            'psnr': final_psnr,
            'ssim': final_ssim,
            'dataset': dataset_name
        }

    # TODO modify this function to configurate the optimizer of your pipeline
    def __setup_optimizers__(self):
        train_opt = self.config['optim_config'] 
        optim_params = []
        for k, v in self.network.named_parameters():
            if v.requires_grad:
                optim_params.append(v)
            else:
                self.reporter.writeInfo(f'Params {k} will not be optimized.')

        optim_type = self.config['optim_type']
        if optim_type.lower() == 'adam':
            self.optimizer = torch.optim.Adam(optim_params,**train_opt)
        elif optim_type.lower() == 'adamw':
            self.optimizer = torch.optim.AdamW(optim_params,**train_opt)
        else:
            raise NotImplementedError(
                f'optimizer {optim_type} is not supperted yet.')
        
        # 改进的学习率调度器配置
        # 使用更稳定的学习率调度策略
        from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, ReduceLROnPlateau
        
        # 选择调度器类型
        scheduler_type = self.config.get('scheduler_type', 'cosine')
        
        if scheduler_type == 'cosine':
            # 使用余弦退火，但周期更长，避免学习率下降过快
            T_0 = self.config.get('scheduler_T_0', 100)  # 第一次重启的周期
            T_mult = self.config.get('scheduler_T_mult', 2)  # 每次重启后周期翻倍
            self.scheduler = CosineAnnealingWarmRestarts(
                self.optimizer, 
                T_0=T_0, 
                T_mult=T_mult,
                eta_min=self.config.get('scheduler_eta_min', 1e-6)
            )
            print(f"使用 CosineAnnealingWarmRestarts 学习率调度器 (T_0={T_0}, T_mult={T_mult})")
        elif scheduler_type == 'plateau':
            # 基于验证性能的学习率调度
            self.scheduler = ReduceLROnPlateau(
                self.optimizer,
                mode='max',  # 监控PSNR，越大越好
                factor=0.5,
                patience=10,
                verbose=True,
                min_lr=1e-6
            )
            print("使用 ReduceLROnPlateau 学习率调度器")
        else:
            # 默认使用简单的余弦退火
            self.scheduler = CosineAnnealingWarmRestarts(
                self.optimizer, 
                T_0=50,
                eta_min=1e-6
            )
            print("使用默认 CosineAnnealingWarmRestarts 学习率调度器")
        
        # 记录初始学习率
        initial_lr = self.optimizer.param_groups[0]['lr']
        print(f"初始学习率: {initial_lr}")
        self.reporter.writeInfo(f"Initial learning rate: {initial_lr}")
        
        # self.optimizers.append(self.optimizer_g)
        

    def train(self):
        
        # general configurations 
        ckpt_dir    = self.config["project_checkpoints"]
        log_frep    = self.config["log_step"]
        model_freq  = self.config["model_save_epoch"]
        total_epoch = self.config["total_epoch"]
        l1_W        = self.config["l1_weight"]
        # lrDecayStep = self.config["lrDecayStep"]
        # TODO [more configurations here]
        self.best_psnr = {
            "epoch":-1.0,
            "psnr":-1.0
        }

        #===============build framework================#
        self.__init_framework__()
        
        # 计算模型复杂度
        from thop import profile
        from thop import clever_format
        train_patch_size = self.config["dataset_params"]["lr_patch_size"]
        
        # 创建测试张量并确保与模型在同一设备上
        if isinstance(self.network, nn.DataParallel):
            # 对于 DataParallel 模型，使用 device_ids[0]
            test_device = next(self.network.parameters()).device
        else:
            test_device = torch.device('cuda' if torch.cuda.is_available() and not self.config.get("use_cpu", False) else 'cpu')
        
        test_img = torch.rand((1, 3, train_patch_size, train_patch_size)).to(test_device)

        # 临时禁用 DataParallel 进行 FLOPs 计算（避免设备不一致问题）
        if isinstance(self.network, nn.DataParallel):
            original_model = self.network.module
        else:
            original_model = self.network
        
        macs, params = profile(original_model, inputs=(test_img,))
        macs, params = clever_format([macs, params], "%.3f")
        print("Model FLOPs: ", macs)
        print("Model Params:", params)
        self.reporter.writeInfo("Model FLOPs: " + macs)
        self.reporter.writeInfo("Model Params: " + params)

        # set the start point for training loop
        if self.config.get("resume", False) and self.config.get("resume_epoch", 0) > 0:
            start = self.config["resume_epoch"]  # 从resume_epoch开始训练
            print(f"Resume training from epoch {start}")
        elif self.config["phase"] == "finetune":
            start = self.config["checkpoint_epoch"] - 1
        else:
            start = 0
        

        #===============build optimizer================#
        print("build the optimizer...")
        # Optimizer
        # TODO replace below lines to build your optimizer
        self.__setup_optimizers__()

        #===============build losses===================#
        # TODO replace below lines to build your losses
        l1 = nn.L1Loss() # [replace this]
        
        # 确保损失函数在正确的设备上
        if not self.config.get("use_cpu", False):
            l1 = l1.cuda()

        # Caculate the epoch number
        step_epoch  = len(self.train_loader)
        print("Total step = %d in each epoch"%step_epoch)

        # Start time
        import datetime
        print("Start to train at %s"%(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
        
        print('Start   ===========================  training...')
        start_time = time.time()
        # import pdb; pdb.set_trace()
        for epoch in range(start, total_epoch):
            epoch_start_time = time.time()  # 记录epoch开始时间
            step = 0
            for hr, lr in self.train_loader:
                step += 1
                # Set the networks to train mode

                self.network.train()
                # TODO [add more code here]
                # clear cumulative gradient
                self.optimizer.zero_grad()

                # TODO read the training data
                if not self.config.get("use_cpu", False):
                    hr = hr.cuda(non_blocking=True)
                    lr = lr.cuda(non_blocking=True)
                else:
                    hr = hr.cpu()
                    lr = lr.cpu()
                
                # 修复：移除重复归一化，数据已经在dataloader中通过ToTensor()和Normalize()处理过了
                # hr = (hr / 255.0 - 0.5) * 2.0
                # lr = (lr / 255.0 - 0.5) * 2.0

                # TODO forward pass
                res = self.network(lr)
                
                # TODO calculate loss
                loss_l1 = l1(res, hr)
                loss_curr = l1_W * loss_l1
                
                # 添加梯度裁剪以防止梯度爆炸
                loss_curr.backward()
                
                # 梯度裁剪
                max_grad_norm = self.config.get('max_grad_norm', 1.0)
                if max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_grad_norm)
                
                # 检查梯度是否正常
                grad_norm = 0.0
                for param in self.network.parameters():
                    if param.grad is not None:
                        grad_norm += param.grad.data.norm(2).item() ** 2
                grad_norm = grad_norm ** 0.5
                
                # 如果梯度异常，跳过这次更新
                if torch.isnan(grad_norm) or torch.isinf(grad_norm) or grad_norm > 1000:
                    print(f"警告: 梯度异常 (norm={grad_norm:.4f})，跳过更新")
                    self.optimizer.zero_grad()
                    continue
                
                self.optimizer.step()

                # Print out log info
                if (step + 1) % log_frep == 0:
                    elapsed = time.time() - start_time
                    elapsed = str(datetime.timedelta(seconds=elapsed))

                    # cumulative steps
                    cum_step = (step_epoch * epoch + step + 1)
                    
                    #==================Print log info======================#
                    print("[{}], Elapsed [{}], Epoch [{}/{}], Step [{}/{}], loss: {:.4f}, l1: {:.4f}".
                        format(self.config["version"], elapsed, epoch + 1, total_epoch, step + 1, step_epoch, 
                                loss_curr.item(),loss_l1.item()))
                    
                    #===================Write log info into log file=======#
                    self.reporter.writeTrainLog(epoch+1,step+1,
                                "loss: {:.4f}, l1: {:.4f}".format(loss_curr.item(), loss_l1.item()))

                    #==================Tensorboard=========================#
                    # write training information into tensorboard log files
                    if self.config["use_tensorboard"]:
                        self.tensorboard_writer.add_scalar('data/loss', loss_curr.item(), cum_step)
                        self.tensorboard_writer.add_scalar('data/l1', loss_l1.item(), cum_step)

            
            #===============adjust learning rate============#
            # 改进的学习率调度
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # 根据调度器类型调整学习率
            if hasattr(self.scheduler, 'step'):
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    # 对于ReduceLROnPlateau，需要传入验证指标
                    # 这里使用Urban100的PSNR作为监控指标
                    if hasattr(self, 'last_urban100_psnr'):
                        self.scheduler.step(self.last_urban100_psnr)
                else:
                    # 对于其他调度器，直接调用step()
                    self.scheduler.step()
            
            # 记录学习率变化
            new_lr = self.optimizer.param_groups[0]['lr']
            if new_lr != current_lr:
                print(f"学习率变化: {current_lr:.2e} -> {new_lr:.2e}")
                self.reporter.writeTrainLog(epoch+1, 0, f"Learning rate changed: {current_lr:.2e} -> {new_lr:.2e}")
            
            # 保留原有的手动衰减作为备选方案（但默认禁用）
            if (epoch + 1) in self.config["lr_decay_step"] and self.config.get("lr_decay_enable", False):
                print("手动学习率衰减")
                for p in self.optimizer.param_groups:
                    p['lr'] *= self.config["lr_decay"]
                    print("Current learning rate is %f"%p['lr'])

            #===============每个epoch后进行评估和保存最优模型================#
            print(f"\n🔍 Epoch {epoch+1} 完成，开始评估模型性能...")
            
            # 评估所有数据集
            eval_results = {}
            eval_results['Set5'] = self.__evaluation__(self.eval_loader1, self.eval_iter1, epoch+1)
            eval_results['Set14'] = self.__evaluation__(self.eval_loader2, self.eval_iter2, epoch+1)
            eval_results['B100'] = self.__evaluation__(self.eval_loader3, self.eval_iter3, epoch+1)
            eval_results['Urban100'] = self.__evaluation__(self.eval_loader4, self.eval_iter4, epoch+1)
            eval_results['Manga109'] = self.__evaluation__(self.eval_loader5, self.eval_iter5, epoch+1)
            
            # 计算平均PSNR和SSIM
            avg_psnr = sum(result['psnr'] for result in eval_results.values()) / len(eval_results)
            avg_ssim = sum(result['ssim'] for result in eval_results.values()) / len(eval_results)
            
            print(f"\n📊 Epoch {epoch+1} 平均性能:")
            print(f"   平均PSNR: {avg_psnr:.4f}")
            print(f"   平均SSIM: {avg_ssim:.5f}")
            
            # 保存当前epoch的模型
            if (epoch+1) % model_freq==0:
                print(f"💾 保存Epoch {epoch+1} 模型检查点...")
                torch.save(self.network.state_dict(),
                        os.path.join(ckpt_dir, 'epoch{}_{}.pth'.format(epoch + 1, 
                                    self.config["checkpoint_names"]["generator_name"])))
            
            # 检查是否为最优模型（基于Urban100的PSNR）
            urban100_psnr = eval_results['Urban100']['psnr']
            # 保存Urban100 PSNR用于学习率调度
            self.last_urban100_psnr = urban100_psnr
            
            if urban100_psnr > self.best_psnr["psnr"]:
                self.best_psnr["psnr"] = float(urban100_psnr)
                self.best_psnr["epoch"] = int(epoch + 1)
                
                print(f"\n🏆 发现新的最优模型!")
                print(f"   Urban100 PSNR: {urban100_psnr:.4f} (Epoch {epoch+1})")
                print(f"   平均PSNR: {avg_psnr:.4f}")
                print(f"   平均SSIM: {avg_ssim:.5f}")
                
                # 保存最优模型
                best_model_path = os.path.join(ckpt_dir, 'best_model_{}_epoch{}.pth'.format(
                    self.config["checkpoint_names"]["generator_name"], epoch + 1))
                torch.save(self.network.state_dict(), best_model_path)
                print(f"💾 最优模型已保存: {best_model_path}")
                
                # 保存最优模型的详细信息
                best_info_path = os.path.join(ckpt_dir, 'best_model_info.txt')
                with open(best_info_path, 'w', encoding='utf-8') as f:
                    f.write(f"最优模型信息:\n")
                    f.write(f"Epoch: {epoch+1}\n")
                    f.write(f"Urban100 PSNR: {urban100_psnr:.4f}\n")
                    f.write(f"平均PSNR: {avg_psnr:.4f}\n")
                    f.write(f"平均SSIM: {avg_ssim:.5f}\n")
                    f.write(f"各数据集性能:\n")
                    for dataset, result in eval_results.items():
                        f.write(f"  {dataset}: PSNR={result['psnr']:.4f}, SSIM={result['ssim']:.5f}\n")
                    f.write(f"模型路径: {best_model_path}\n")
                
                print(f"📝 最优模型信息已保存: {best_info_path}")
            else:
                print(f"📈 当前模型性能: Urban100 PSNR={urban100_psnr:.4f} (最优: {self.best_psnr['psnr']:.4f} @ Epoch {self.best_psnr['epoch']})")
            
            print(f"⏱️  Epoch {epoch+1} 评估完成，耗时: {time.time() - epoch_start_time:.2f}秒")
            print("="*80)
