import io
import pandas as pd
import ai_assistant

csv='a,b,c\n1,2,3\n4,5,6\n7,8,9'
df=pd.read_csv(io.StringIO(csv))
print('insights', ai_assistant.dataset_insights(df))
print('columns', ai_assistant.analyze_query(df,'columns'))
print('mean', ai_assistant.analyze_query(df,'mean'))
print('corr', ai_assistant.analyze_query(df,'correlation'))
res = ai_assistant.analyze_query(df,'plot a')
print('plot result', res)
