def tokenize_inputs(config, tokenizer, examples):
    max_length = config["max_length"]  # 从配置中获取最大长度

    # hacky backward compatible
    different_eos = tokenizer.eos_token != "</s>"  # 检查分词器的eos token是否与标准的不同
    out = {"labels": [], "input_ids": [], "attention_mask": []}  # 初始化输出字典
    for prompt, response in zip(examples["prompt"], examples["response"]):  # 遍历每个提示和响应
        if different_eos:
            if response.count("</s> \n") > 0:
                response = response.replace("</s> \n", f"{tokenizer.eos_token} \n")  # 替换响应中的结束标记

        prompt_len = len(tokenizer(prompt + "\n", return_tensors="pt")["input_ids"][0])  # 获取提示的长度

        # hack if our prompt is super long
        # 如果提示过长，我们需要截断它以确保包含一些标签
        if prompt_len >= max_length // 2:
            # 如果提示过长，进行截断，但确保不超过1024个标记
            new_len = min(max_length // 2, len(prompt) // 2)
            prompt = prompt[:new_len]
            # 获取新的提示长度
            prompt_len = tokenizer(prompt + "\n", return_tensors="pt", max_length=max_length // 2, truncation=True).input_ids.ne(tokenizer.pad_token_id).sum().item()

        assert prompt_len <= max_length // 2, f"prompt length {prompt_len} exceeds max length {max_length}"  # 确认提示长度不超过最大长度的一半

        input_tokens = tokenizer(prompt + "\n" + response + tokenizer.eos_token,
                                 truncation=True, max_length=max_length, return_tensors="pt")["input_ids"].squeeze()  # 获取输入标记序列

        labels = input_tokens.clone()  # 克隆输入标记序列作为标签
        labels[:prompt_len] = -100  # 将提示部分的标签设为-100，以忽略其损失计算
        if len(labels) < max_length:
            # 将标签填充到最大长度，填充值为-100
            labels = torch.cat([labels, torch.full((max_length - len(labels),), -100)])

        assert (labels == -100).sum() < len(labels), f"Labels are all -100, something wrong. prompt length {prompt_len} exceeds max length {max_length}"  # 确认标签不全是-100
        
        if (labels == -100).sum() == len(labels) - 1:
            print(prompt)
            print(response)
            raise

        padded = tokenizer.pad({"input_ids": input_tokens}, padding="max_length", max_length=max_length, return_tensors="pt")  # 填充输入标记序列
        out["labels"].append(labels)  # 将标签添加到输出字典中
        out["input_ids"].append(padded["input_ids"])  # 将输入标记序列添加到输出字典中
        out["attention_mask"].append(padded["attention_mask"])  # 将注意力掩码添加到输出字典中

    out = {k: torch.stack(v) if isinstance(v, list) else v for k, v in out.items()}  # 将列表中的张量堆叠成一个张量

    return out  # 返回处理后的输出字典
