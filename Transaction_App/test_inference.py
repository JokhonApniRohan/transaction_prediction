import sys
sys.path.append('../Modules')
import pandas as pd
from utils.inference import load_model, predict_next_day

try:
    model, features, scaler = load_model('model/best_model.pkl')
    df = pd.read_csv('data/new_transaction_data.csv')
    result = predict_next_day(df, model, scaler, features)
    print('✅ SUCCESS!')
    print('   Date:', result['date'])
    print('   Current:', f"{result['current']:,.0f}")
    print('   Delta:', f"{result['delta']:,.0f}")
    print('   Next Day:', f"{result['next_day']:,.0f}")
    print('\n   Dictionary keys:', list(result.keys()))
except Exception as e:
    print('❌ ERROR:', e)
    import traceback
    traceback.print_exc()
