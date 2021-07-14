from rouge import Rouge

def rouge_eval(text, summ):
    rouge = Rouge()
    return rouge.get_scores(text, summ, avg=True)