from tqdm import tqdm
import multiprocessing
from openai import OpenAI
from argparse import ArgumentParser
import logging
import os
import jsonlines


def set_gpt():
    api_key = "sk-JyOqNQKvXozZ0kZR0cBeF7701c2042A29e5b55219304Eb85"
    api_base = "https://api.xi-ai.cn/v1"
    client = OpenAI(api_key=api_key, base_url=api_base)
    return client


def get_prompt(mode, item):
    prompt_open = """
    I need your help to evaluate the performance of several models in the speech interaction scenario. The models will receive a speech input from the user, which they need to understand and respond to with a speech output.
    Your task is to rate the model’s responses based on the provided user input transcription [Instruction] and the model’s output transcription [Response].

    Please evaluate the response on a scale of 1 to 5:
    1 point: The response is largely irrelevant, incorrect, or fails to address the user’s query. It may be off-topic or provide incorrect information.
    2 points: The response is somewhat relevant but lacks accuracy or completeness. It may only partially answer the user’s question or include extraneous information.
    3 points: The response is relevant and mostly accurate, but it may lack conciseness or include unnecessary details that don’t contribute to the main point.
    4 points: The response is relevant, accurate, and concise, providing a clear answer to the user’s question without unnecessary elaboration.
    5 points: The response is exceptionally relevant, accurate, and to the point. It directly addresses the user’s query in a highly effective and efficient manner, providing exactly the information needed.

    Below are the transcription of user’s instruction and models’ response:
    ### [Instruction]
    {question}
    
    ### [Response]
    {answer}

    After evaluating, please output the score only without anything else.
    You don’t need to provide any explanations.
    """.strip()

    prompt_semi_open = """
    I need your help to evaluate the performance of several models in the speech interaction scenario. The models will receive a speech input from the user, which they need to understand and respond to with a speech output.
    Your task is to rate the model’s responses based on the provided user input transcription [Instruction], the model’s output transcription [Response] and some suggested answers [Reference].
    The model's response doesn't necessarily have to be identical to the suggested answers, as long as it aligns with the question and is reasonable.

    Please evaluate the response on a scale of 1 to 5:
    1 point: The response is largely irrelevant, incorrect, or fails to address the user's query. It may be off-topic or provide incorrect information. The response does not align with the question in any meaningful way.
    2 points: The response is somewhat relevant but lacks accuracy, completeness, or coherence. It may partially address the query but introduces unnecessary information or deviates from the core issue. The response may not align well with the suggested answer but still provides some value.
    3 points: The response is relevant and mostly accurate, but may lack conciseness or clarity. It addresses the question reasonably, but there might be slight deviations in approach or content. While it may not strictly align with the suggested answer, it still effectively addresses the core of the query.
    4 points: The response is relevant, accurate, and concise. It provides a clear answer to the user’s question and avoids unnecessary details. While it may not exactly mirror the suggested answer, it effectively addresses the user's query in a logical and well-reasoned manner.
    5 points: The response is exceptionally relevant, accurate, and concise. It directly addresses the user's query in the most efficient manner, providing exactly the information needed. The response may differ from the suggested answer in phrasing or approach but still aligns perfectly with the intent of the query, demonstrating a high level of reasoning and clarity.

    Below are the transcription of user’s instruction, models’ response and the reference answer:
    ### [Instruction]
    {question}

    ### [Response]
    {answer}

    ### [Reference]
    {reference}

    After evaluating, please output the score only without anything else.
    You don’t need to provide any explanations.
    """.strip()

    prompt_qa = """
    I need your help to evaluate the performance of several models in the speech interaction scenario. The models will receive a speech input from the user, which they need to understand and respond to with a speech output.
    Your task is to rate the model’s responses based on the provided user input transcription [Question], the model’s output transcription [Response] and the correct answer [Reference].
    
    Below are the transcription of user’s instruction, models’ response and the reference answer:
    ### [Question]
    {question}

    ### [Response]
    {answer}

    ### [Reference]
    {reference}

    Is the model’s response correct based on the question and reference answer? 
    Please only output a single "Yes" or "No". Do not output anything else.
    """.strip()

    prompt_multi2 = """
    I need your help to evaluate the performance of several models in the multi-round speech interaction scenario. The models will receive a speech input from the user, which they need to understand and respond to with a speech output.
    Your task is to rate the model’s multi-round responses based on the provided user input transcription [Instruction], the model’s output transcription [Response] and some suggested answers [Reference].
    The model's response doesn't necessarily have to be identical to the suggested answers, as long as it aligns with the question and is reasonable.

    Please evaluate the response on a scale of 1 to 5:
    1 point: Responses are irrelevant or nonsensical. Or responses ignore previous turns, leading to confusion or irrelevance.
    2 points: Some answers are relevant but many lack detail or completeness. Frequently loses track of the conversation, with responses that are not aligned with earlier turns.
    3 points: Responses are mostly relevant and coherent, though occasional lapses in depth. The model follows the conversation, but may occasionally forget important details from earlier turns.
    4 points: Responses are clear, relevant, and detailed. Generally keeps track of the conversation, with minor lapses.
    5 points: Responses are clear, relevant, and detailed. Flawlessly integrates context across all rounds, ensuring natural conversation flow, creating an engaging experience.

    Below are the transcription of user’s instruction, models’ response and the reference answer:
    ### [Round_1]
    ### [Instruction]
    {question1}
    ### [Response]
    {answer1}
    ### [Reference]
    {reference1}

    ### [Round_2]
    ### [Instruction]
    {question2}
    ### [Response]
    {answer2}
    ### [Reference]
    {reference2}

    Please output only one score for the whole conversation without anything else.
    You don’t need to provide any explanations.
    """.strip()

    prompt_multi3 = """
    I need your help to evaluate the performance of several models in the multi-round speech interaction scenario. The models will receive a speech input from the user, which they need to understand and respond to with a speech output.
    Your task is to rate the model’s multi-round responses based on the provided user input transcription [Instruction], the model’s output transcription [Response] and some suggested answers [Reference].
    The model's response doesn't necessarily have to be identical to the suggested answers, as long as it aligns with the question and is reasonable.

    Please evaluate the response on a scale of 1 to 5:
    1 point: Responses are irrelevant or nonsensical. Or responses ignore previous turns, leading to confusion or irrelevance.
    2 points: Some answers are relevant but many lack detail or completeness. Frequently loses track of the conversation, with responses that are not aligned with earlier turns.
    3 points: Responses are mostly relevant and coherent, though occasional lapses in depth. The model follows the conversation, but may occasionally forget important details from earlier turns.
    4 points: Responses are clear, relevant, and detailed. Generally keeps track of the conversation, with minor lapses.
    5 points: Responses are clear, relevant, and detailed. Flawlessly integrates context across all rounds, ensuring natural conversation flow, creating an engaging experience.

    Below are the transcription of user’s instruction, models’ response and the reference answer:
    ### [Round_1]
    ### [Instruction]
    {question1}
    ### [Response]
    {answer1}
    ### [Reference]
    {reference1}

    ### [Round_2]
    ### [Instruction]
    {question2}
    ### [Response]
    {answer2}
    ### [Reference]
    {reference2}

    ### [Round_3]
    ### [Instruction]
    {question3}
    ### [Response]
    {answer3}
    ### [Reference]
    {reference3}

    Please output only one score for the whole conversation without anything else.
    You don’t need to provide any explanations.
    """.strip()

    prompt_multi4 = """
    I need your help to evaluate the performance of several models in the multi-round speech interaction scenario. The models will receive a speech input from the user, which they need to understand and respond to with a speech output.
    Your task is to rate the model’s multi-round responses based on the provided user input transcription [Instruction], the model’s output transcription [Response] and some suggested answers [Reference].
    The model's response doesn't necessarily have to be identical to the suggested answers, as long as it aligns with the question and is reasonable.

    Please evaluate the response on a scale of 1 to 5:
    1 point: Responses are irrelevant or nonsensical. Or responses ignore previous turns, leading to confusion or irrelevance.
    2 points: Some answers are relevant but many lack detail or completeness. Frequently loses track of the conversation, with responses that are not aligned with earlier turns.
    3 points: Responses are mostly relevant and coherent, though occasional lapses in depth. The model follows the conversation, but may occasionally forget important details from earlier turns.
    4 points: Responses are clear, relevant, and detailed. Generally keeps track of the conversation, with minor lapses.
    5 points: Responses are clear, relevant, and detailed. Flawlessly integrates context across all rounds, ensuring natural conversation flow, creating an engaging experience.

    Below are the transcription of user’s instruction, models’ response and the reference answer:
    ### [Round_1]
    ### [Instruction]
    {question1}
    ### [Response]
    {answer1}
    ### [Reference]
    {reference1}

    ### [Round_2]
    ### [Instruction]
    {question2}
    ### [Response]
    {answer2}
    ### [Reference]
    {reference2}

    ### [Round_3]
    ### [Instruction]
    {question3}
    ### [Response]
    {answer3}
    ### [Reference]
    {reference3}

    ### [Round_4]
    ### [Instruction]
    {question4}
    ### [Response]
    {answer4}
    ### [Reference]
    {reference4}

    Please output only one score for the whole conversation without anything else.
    You don’t need to provide any explanations.
    """.strip()

    prompt_multi5 = """
    I need your help to evaluate the performance of several models in the multi-round speech interaction scenario. The models will receive a speech input from the user, which they need to understand and respond to with a speech output.
    Your task is to rate the model’s multi-round responses based on the provided user input transcription [Instruction], the model’s output transcription [Response] and some suggested answers [Reference].
    The model's response doesn't necessarily have to be identical to the suggested answers, as long as it aligns with the question and is reasonable.

    Please evaluate the response on a scale of 1 to 5:
    1 point: Responses are irrelevant or nonsensical. Or responses ignore previous turns, leading to confusion or irrelevance.
    2 points: Some answers are relevant but many lack detail or completeness. Frequently loses track of the conversation, with responses that are not aligned with earlier turns.
    3 points: Responses are mostly relevant and coherent, though occasional lapses in depth. The model follows the conversation, but may occasionally forget important details from earlier turns.
    4 points: Responses are clear, relevant, and detailed. Generally keeps track of the conversation, with minor lapses.
    5 points: Responses are clear, relevant, and detailed. Flawlessly integrates context across all rounds, ensuring natural conversation flow, creating an engaging experience.

    Below are the transcription of user’s instruction, models’ response and the reference answer:
    ### [Round_1]
    ### [Instruction]
    {question1}
    ### [Response]
    {answer1}
    ### [Reference]
    {reference1}

    ### [Round_2]
    ### [Instruction]
    {question2}
    ### [Response]
    {answer2}
    ### [Reference]
    {reference2}

    ### [Round_3]
    ### [Instruction]
    {question3}
    ### [Response]
    {answer3}
    ### [Reference]
    {reference3}

    ### [Round_4]
    ### [Instruction]
    {question4}
    ### [Response]
    {answer4}
    ### [Reference]
    {reference4}

    ### [Round_5]
    ### [Instruction]
    {question5}
    ### [Response]
    {answer5}
    ### [Reference]
    {reference5}

    Please output only one score for the whole conversation without anything else.
    You don’t need to provide any explanations.
    """.strip()

    prompt_contrast = """
    I need your help to compare the performance of two models in the speech interaction scenario. The two models will receive the same speech input from the user, which they need to understand and respond to with a speech output.
    Your task is to compare the two models’ responses based on the provided user input transcription [Question] and the two models’ output transcription [Answer_1] and [Answer_2].
    You need to evaluate the outputs of the two models comprehensively based on their Relevance, Accuracy, Clarity, and Completeness.

    Please provide a score between -2 and 2 based on the following criteria:
    -2 point: [Answer_1] is significantly worse than [Answer_2].
    -1 points: [Answer_1] is slightly worse than [Answer_2].
    0 points: [Answer_1] and [Answer_2] are equally good or bad.
    1 points: [Answer_1] is slightly better than [Answer_2].
    2 points: [Answer_1] is significantly better than [Answer_2].

    Below are the transcription of user’s instruction and models’ response:
    ### Question
    {question}

    ### Answer_1
    {answer_1}

    ### Answer_2
    {answer_2}

    After evaluating, please output the score only without anything else.
    You don’t need to provide any explanations.
    """.strip()

    if mode == "open":
        return prompt_open.replace("{question}", item["question"]).replace(
            "{answer}", item["answer"]
        )
    elif mode == "semi-open":
        return (
            prompt_semi_open.replace("{question}", item["question"])
            .replace("{answer}", item["answer"])
            .replace("{reference}", item["reference"])
        )
    elif mode == "qa":
        return (
            prompt_qa.replace("{question}", item["question"])
            .replace("{answer}", item["answer"])
            .replace("{reference}", item["reference"])
        )
    elif mode == "multi":
        if item["round"] == 2:
            return (
                prompt_multi2.replace("{question1}", item["question0"])
                .replace("{question2}", item["question1"])
                .replace("{answer1}", item["answer0"])
                .replace("{answer2}", item["answer1"])
                .replace("{reference1}", item["reference0"])
                .replace("{reference2}", item["reference1"])
            )
        elif item["round"] == 3:
            return (
                prompt_multi3.replace("{question1}", item["question0"])
                .replace("{question2}", item["question1"])
                .replace("{question3}", item["question2"])
                .replace("{answer1}", item["answer0"])
                .replace("{answer2}", item["answer1"])
                .replace("{answer3}", item["answer2"])
                .replace("{reference1}", item["reference0"])
                .replace("{reference2}", item["reference1"])
                .replace("{reference3}", item["reference2"])
            )
        elif item["round"] == 4:
            return (
                prompt_multi4.replace("{question1}", item["question0"])
                .replace("{question2}", item["question1"])
                .replace("{question3}", item["question2"])
                .replace("{question4}", item["question3"])
                .replace("{answer1}", item["answer0"])
                .replace("{answer2}", item["answer1"])
                .replace("{answer3}", item["answer2"])
                .replace("{answer4}", item["answer3"])
                .replace("{reference1}", item["reference0"])
                .replace("{reference2}", item["reference1"])
                .replace("{reference3}", item["reference2"])
                .replace("{reference4}", item["reference3"])
            )
        elif item["round"] == 5:
            return (
                prompt_multi5.replace("{question1}", item["question0"])
                .replace("{question2}", item["question1"])
                .replace("{question3}", item["question2"])
                .replace("{question4}", item["question3"])
                .replace("{question5}", item["question4"])
                .replace("{answer1}", item["answer0"])
                .replace("{answer2}", item["answer1"])
                .replace("{answer3}", item["answer2"])
                .replace("{answer4}", item["answer3"])
                .replace("{answer5}", item["answer4"])
                .replace("{reference1}", item["reference0"])
                .replace("{reference2}", item["reference1"])
                .replace("{reference3}", item["reference2"])
                .replace("{reference4}", item["reference3"])
                .replace("{reference5}", item["reference4"])
            )
    elif mode == "contrast":
        return (
            prompt_contrast.replace("{question}", item["question"])
            .replace("{answer_1}", item["answer_1"])
            .replace("{answer_2}", item["answer_2"])
        )


def mark(prompt, client):
    try:
        scores = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=1024,
            frequency_penalty=0,
            presence_penalty=0,
            stop=None,
            temperature=0.5,
            top_p=0.95,
            n=3,
        )
    except:
        return [0, 0, 0]
    else:
        return [choice.message.content for choice in scores.choices]


def eval(args):
    client = set_gpt()
    logging.info("<------start GPT eval------>")

    if args.mode == "open":
        output_file = os.path.join(args.output_dir, "result_open.jsonl")
        sum_score = 0
        with open(args.question, "r") as f:
            length = sum([1 for _ in f])
        with open(args.question, "r") as qt, open(
            args.answer, "r"
        ) as pt, jsonlines.open(output_file, mode="w") as ot:
            for i, (question, answer) in tqdm(
                enumerate(zip(jsonlines.Reader(qt), jsonlines.Reader(pt))), total=length
            ):
                item = {
                    "question": question[str(i).zfill(4)],
                    "answer": answer[str(i).zfill(4)],
                }
                prompt = get_prompt(args.mode, item)
                scores = mark(prompt, client)
                item["score"] = scores
                ot.write(item)
                sum_score += sum([int(i) for i in item["score"]]) / len(item["score"])
            ot.write({"final_score": sum_score * 20 / length})
        # save results
        logging.info(f"saving result to {output_file}")

    elif args.mode == "semi-open":
        output_file = os.path.join(args.output_dir, "result_semi_open.jsonl")
        sum_score = 0
        with open(args.question, "r") as f:
            length = sum([1 for _ in f])
        with open(args.question, "r") as qt, open(args.answer, "r") as pt, open(
            args.reference, "r"
        ) as gt, jsonlines.open(output_file, mode="w") as ot:
            for i, (question, answer, reference) in tqdm(
                enumerate(
                    zip(
                        jsonlines.Reader(qt), jsonlines.Reader(pt), jsonlines.Reader(gt)
                    )
                ),
                total=length,
            ):
                item = {
                    "question": question[str(i).zfill(4)],
                    "answer": answer[str(i).zfill(4)],
                    "reference": reference[str(i).zfill(4)],
                }
                prompt = get_prompt(args.mode, item)
                scores = mark(prompt, client)
                item["score"] = scores
                ot.write(item)
                sum_score += sum([int(i) for i in item["score"]]) / len(item["score"])
            ot.write({"final_score": sum_score * 20 / length})
        # save results
        logging.info(f"saving result to {output_file}")

    elif args.mode == "qa":
        output_file = os.path.join(args.output_dir, "result_qa.jsonl")
        sum_score = 0
        with open(args.question, "r") as f:
            length = sum([1 for _ in f])
        with open(args.question, "r") as qt, open(args.answer, "r") as pt, open(
            args.reference, "r"
        ) as gt, jsonlines.open(output_file, mode="w") as ot:
            for i, (question, answer, reference) in tqdm(
                enumerate(
                    zip(
                        jsonlines.Reader(qt), jsonlines.Reader(pt), jsonlines.Reader(gt)
                    )
                ),
                total=length,
            ):
                item = {
                    "question": question[str(i).zfill(4)],
                    "answer": answer[str(i).zfill(4)],
                    "reference": reference[str(i).zfill(4)],
                }
                prompt = get_prompt(args.mode, item)
                scores = mark(prompt, client)
                item["score"] = scores
                ot.write(item)
                sum_score += sum(
                    [(1 if i == "Yes" else 0) for i in item["score"]]
                ) / len(item["score"])
            ot.write({"final_score": sum_score * 100 / length})
        # save results
        logging.info(f"saving result to {output_file}")

    elif args.mode == "multi":
        output_file = os.path.join(args.output_dir, "result_multi.jsonl")
        sum_score = 0
        with open(args.question, "r") as f:
            length = sum([1 for _ in f])
        with open(args.answer, "r") as pt, jsonlines.open(output_file, mode="w") as ot:
            for i, item in tqdm(
                enumerate(jsonlines.Reader(pt)),
                total=length,
            ):
                question = ["" for _ in range(5)]
                answer = ["" for _ in range(5)]
                reference = ["" for _ in range(5)]
                for j in range(item["num_round"]):
                    question[j] = item["dialogue"][j]["source_text"]
                    answer[j] = item["dialogue"][j]["output_text"]
                    reference[j] = item["dialogue"][j]["target_text"]
                tmp = {"round": item["num_round"]}
                for j in range(5):
                    tmp["question" + str(j)] = question[j]
                    tmp["answer" + str(j)] = answer[j]
                    tmp["reference" + str(j)] = reference[j]
                prompt = get_prompt(args.mode, tmp)
                scores = mark(prompt, client)
                tmp["score"] = scores
                ot.write(tmp)
                sum_score += sum([int(i) for i in tmp["score"]]) / len(tmp["score"])
            ot.write({"final_score": sum_score * 20 / length})
        # save results
        logging.info(f"saving result to {output_file}")

    elif args.mode == "contrast":
        output_file = os.path.join(args.output_dir, "result_contrast.jsonl")
        sum_score = 0
        with open(args.question, "r") as f:
            length = sum([1 for _ in f])
        with open(args.question, "r") as qt, open(args.answer, "r") as pt, open(
            args.answer_contrast, "r"
        ) as ct, jsonlines.open(output_file, mode="w") as ot:
            for i, (question, answer, answer_contrast) in tqdm(
                enumerate(
                    zip(
                        jsonlines.Reader(qt), jsonlines.Reader(pt), jsonlines.Reader(ct)
                    )
                ),
                total=length,
            ):
                item = {
                    "question": question[str(i).zfill(4)],
                    "answer_1": answer[str(i).zfill(4)],
                    "answer_2": answer_contrast[str(i).zfill(4)],
                }
                prompt = get_prompt(args.mode, item)
                scores = mark(prompt, client)
                item["score"] = scores
                ot.write(item)
                sum_score += sum([int(i) for i in item["score"]]) / len(item["score"])
            ot.write({"final_score": sum_score * 50 / length})
        # save results
        logging.info(f"saving result to {output_file}")
