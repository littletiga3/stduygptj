def train(accelerator, config):
    set_seed(config['seed'])  # 设置随机种子以确保结果可复现

    accelerator.print(config)  # 打印配置文件
    accelerator.print(f"Using {accelerator.num_processes} GPUs")  # 打印使用的GPU数量

    # 初始化分词器
    tokenizer = AutoTokenizer.from_pretrained(config['tokenizer_name'], model_max_length=config['max_length'], use_fast=False)
    
    # 如果没有pad token，则将其设置为eos token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    with accelerator.main_process_first():
        train_dataloader, val_dataloader = load_data(config, tokenizer)  # 加载训练和验证数据

    checkpoint = config["gradient_checkpointing"]

    # 加载模型
    model = AutoModelForCausalLM.from_pretrained(config["model_name"], 
                                                 use_cache=False if checkpoint else True,
                                                 trust_remote_code=True) 
    if checkpoint:
        model.gradient_checkpointing_enable()  # 如果启用了梯度检查点，则启用相应设置

    if config["lora"]:
        # 配置并应用LoRA
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1
        )
        model = get_peft_model(model, peft_config)  # 获取PEFT模型
        model.print_trainable_parameters()  # 打印可训练参数

    # 确定优化器类
    optimizer_cls = (
        AdamW
        if accelerator.state.deepspeed_plugin is None
        or "optimizer" not in accelerator.state.deepspeed_plugin.deepspeed_config
        else DummyOptim
    )

    # 创建优化器
    optimizer = optimizer_cls(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])

    if accelerator.state.deepspeed_plugin is not None:
        gradient_accumulation_steps = accelerator.state.deepspeed_plugin.deepspeed_config["gradient_accumulation_steps"]

    # 计算训练总步数
    lr_ratio = config["min_lr"] / config["lr"]
    accelerator.print(f"Len of train_dataloader: {len(train_dataloader)}")
    total_num_steps = (len(train_dataloader) / gradient_accumulation_steps) * (config["num_epochs"])
    total_num_steps += int(total_num_steps * lr_ratio) + config["warmup_steps"]
    accelerator.print(f"Total training steps: {total_num_steps}")

    # 创建学习率调度器
    if (
        accelerator.state.deepspeed_plugin is None
        or "scheduler" not in accelerator.state.deepspeed_plugin.deepspeed_config
    ):
        scheduler = get_scheduler(
            name="cosine",
            optimizer=optimizer,
            num_warmup_steps=config["warmup_steps"] * accelerator.num_processes,
            num_training_steps=total_num_steps,
        )
    else:
        scheduler = DummyScheduler(optimizer, total_num_steps=total_num_steps, warmup_num_steps=config["warmup_steps"])

    # 准备模型、优化器、数据加载器和调度器进行训练
    model, optimizer, train_dataloader, val_dataloader, scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, val_dataloader, scheduler
    )

    # 注册调度器以进行检查点保存
    accelerator.register_for_checkpointing(scheduler)

    if config["checkpoint"]:
        # 如果指定了检查点，则从检查点加载
        accelerator.load_state(config["checkpoint"])
        accelerator.print(f"Resumed from checkpoint: {config['checkpoint']}")
        path = os.path.basename(config["checkpoint"])
        training_difference = os.path.splitext(path)[0]
        resume_step = int(training_difference.replace("step_", ""))
        train_dataloader = accelerator.skip_first_batches(train_dataloader, resume_step)
        accelerator.print(f"Resuming from step {resume_step}")
    else:
        resume_step = 0

    # 如果使用wandb，则记录梯度
    if accelerator.is_main_process and config["wandb"]:
        wandb.watch(model, log_freq=config["log_grads_every"], log="all")

    accelerator.wait_for_everyone()

    for epoch in range(0, config["num_epochs"]):
        train_loss = MeanMetric(nan_strategy="error").to(model.device)  # 初始化训练损失的平均指标
        for step, batch in enumerate(tqdm(train_dataloader)):
            curr_step = epoch * len(train_dataloader) + step
            model.train()
            outputs = model(**batch)
            loss = outputs.loss

            # 在反向传播之前收集损失，以便进行梯度累积
            loss_values = accelerator.gather_for_metrics({"loss": loss.detach().float()})
            if config["wandb"]:
                accelerator.log({"loss": torch.mean(loss_values["loss"]).item()}, step=curr_step)
            train_loss.update(loss_values["loss"])

            loss = loss / gradient_accumulation_steps
            accelerator.backward(loss)

            # 记录学习率
            if step > 0 and step % (config["log_lr_every"]) == 0:
                if config["wandb"]:
                    accelerator.log({"lr": scheduler.get_last_lr()[0]}, step=curr_step)

            if (step + 1) % gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            if step > 0 and step % config["save_every"] == 0:
                accelerator.save_state(f"{config['output_dir']}/step_{curr_step}")

            if step > 0 and (step % config["eval_every"] == 0 or step == len(train_dataloader) - 1):
                val_loss = evaluate(model, val_dataloader)

                log_train = {"train_loss": train_loss.compute()}
                log_val = {"val_loss": val_loss.compute()}

                if config["wandb"]:
                    accelerator.log({**log_train, **log_val}, step=curr_step)

                accelerator.print(f"Current LR: {scheduler.get_last_lr()[0]}")
                accelerator.print(format_metrics(log_train, "train", f" step {step} "))
                accelerator.print(format_metrics(log_val, "val", f" step {step} "))

                train_loss.reset()

        accelerator.print(f"Epoch {epoch} finished")
        accelerator.print(f"Pushing to HF hub")
        unwrapped_model = accelerator.unwrap_model(model)

        # 保存模型
        unwrapped_model.save_pretrained(
            f"{config['output_dir']}/epoch_{epoch}",
            is_main_process=accelerator.is_main_process,
            save_function=accelerator.save,
            state_dict=accelerator.get_state_dict(model),
        )
        try:
            if accelerator.is_main_process:
                unwrapped_model.push_to_hub(config["save_name"] + f"-epoch_{epoch}", private=True)
        except Exception as e:
            accelerator.print(e)
            accelerator.print(f"Failed to push to hub")

    if config["num_epochs"] > 1:
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(
            f"{config['output_dir']}/final",
            is_main_process=accelerator.is_main_process,
            save_function=accelerator.save,
            state_dict=accelerator.get_state_dict(model),
        )

    accelerator.end_training()  # 结束训练过程


if __name__ == "__main__":
    # 通过读取配置文件解析参数
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")

    args = parser.parse_args()

    config = read_config(args.config)  # 从指定文件中读取配置

    if config["wandb"]:
        # 使用wandb日志初始化加速器
        accelerator = Accelerator(log_with="wandb")
        accelerator.init_trackers(
            project_name=config["wandb_project_name"],
            config=config,
            init_kwargs={"wandb": {"entity": config["wandb_entity"]}},
        )
    else:
        accelerator = Accelerator()  # 初始化加速器而不使用wandb日志

    train(accelerator, config=config)  # 使用加速器和配置调用训练函数
