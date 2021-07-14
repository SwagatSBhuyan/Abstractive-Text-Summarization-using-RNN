
from transformers import pipeline
summarizer = pipeline("summarization")

def trans_summarizer(text, original_summary):

  start_t = time.perf_counter()

  dicto['Predicted_summary'].append(summarizer(text, max_length=20, min_length=5, do_sample=False)[0]['summary_text'])

  # embedding1 = sentence_model.encode(original_summ, convert_to_tensor=True)
  # embedding2 = sentence_model.encode(predicted_summ, convert_to_tensor=True)
  # cosine_scores = util.pytorch_cos_sim(embedding1, embedding2)

  # rouge_score = rouge.get_scores(original_summ, predicted_summ,avg=True)

  dicto['Text'].append(text)
  dicto['Original_summary'].append(original_summary)
  # dicto['Semantic_similarity'].append(cosine_scores.item())
  # dicto['bleu'].append(nltk.translate.bleu_score.sentence_bleu(original_summ, predicted_summ))
  # dicto['rouge_1_f'].append(rouge_score['rouge-1']['f'])
  # dicto['rouge_2_f'].append(rouge_score['rouge-2']['f'])
  # dicto['rouge_l_f'].append(rouge_score['rouge-l']['f'])
  end_t = time.perf_counter()

  print(f"Summarizer ran in {end_t-start_t} seconds.")