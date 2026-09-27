from datasets import load_dataset


DATASET_NAME = "framolfese/2WikiMultihopQA"


def load_benchmark(split="validation", limit=None):

    dataset = load_dataset(
        DATASET_NAME,
        split=split
    )

    if limit is not None:
        dataset = dataset.select(range(min(limit, len(dataset))))

    return dataset


def normalize_example(example):

    gold_evidence = []

    context = {
        title: sentences
        for title, sentences in zip(
            example["context"]["title"],
            example["context"]["sentences"]
        )
    }

    titles = example["supporting_facts"]["title"]
    sentence_ids = example["supporting_facts"]["sent_id"]

    for title, sentence_id in zip(titles, sentence_ids):

        sentences = context.get(title)

        if sentences is None:
            continue

        if sentence_id >= len(sentences):
            continue

        gold_evidence.append(
            {
                "title": title,
                "sentence_id": sentence_id,
                "text": sentences[sentence_id]
            }
        )

    return {
        "id": example["id"],
        "question": example["question"],
        "gold_answer": example["answer"],
        "gold_evidence": gold_evidence,
        "type": example["type"]
    }


def load_benchmark_examples(split="validation", limit=None):

    dataset = load_benchmark(
        split=split,
        limit=limit
    )

    return [
        normalize_example(example)
        for example in dataset
    ]


if __name__ == "__main__":

    # dataset = load_benchmark(
    #     split="validation",
    #     limit=1
    # )

    # example = dataset[0]

    # print("CONTEXT TYPE:")
    # print(type(example["context"]))

    # print("\nCONTEXT:")
    # print(example["context"])

    # print("\nSUPPORTING FACTS:")
    # print(example["supporting_facts"])

    examples = load_benchmark_examples(
    split="validation",
    limit=1
    )
    print(examples[0]["question"])