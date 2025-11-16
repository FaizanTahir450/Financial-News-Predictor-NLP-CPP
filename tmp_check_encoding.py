import pandas as pd
encodings = ['utf-8','utf-8-sig','latin1','cp1252','iso-8859-1']
path = 'data/financial_news.csv'
for e in encodings:
    try:
        df = pd.read_csv(path, encoding=e, on_bad_lines='skip', engine='python')
        if 'news' in df.columns and 'label' in df.columns:
            print('ok header', e, df.shape)
            break
        df2 = pd.read_csv(path, encoding=e, header=None, names=['label','news'], on_bad_lines='skip', engine='python')
        print('ok noheader', e, df2.shape)
        break
    except Exception as ex:
        print('fail', e, ex)
