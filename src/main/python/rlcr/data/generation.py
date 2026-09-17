def make_generation_dataset(dataset, sys_prompt):
    def make_generation_conversation(example):
        if "question" in example.keys():
            user_format = f"\n\nPROBLEM: {example['question']}\n\n"
        else:
            user_format = f"\n\nPROBLEM: {example['problem']}\n\n"
        return {
            "prompt": [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_format},
            ],
        }

    dataset = dataset.map(make_generation_conversation)
    return dataset
