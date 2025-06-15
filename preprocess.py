from preprocess import preprocess_swat, preprocess_wadi

if __name__ == '__main__':
    original_data_path = ('data/original/swat/SWaT_Dataset_Normal_v1.xlsx', 'data/original/swat/SWaT_Dataset_Attack_v0.xlsx')
    processed_data_path = ('data/processed/swat/train.csv', 'data/processed/swat/test.csv')
    preprocess_swat(original_data_path, processed_data_path)

    original_data_path = ('data/original/wadi/WADI_14days_new.csv', 'data/original/wadi/WADI_attackdataLABLE.csv')
    processed_data_path = ('data/processed/wadi/train.csv', 'data/processed/wadi/test.csv')
    preprocess_wadi(original_data_path, processed_data_path)
