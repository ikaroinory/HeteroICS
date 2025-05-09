from preprocess import preprocess_swat

if __name__ == '__main__':
    swat_list = [
        ('data/original/swat/SWaT_Dataset_Normal_v1.xlsx', 'data/processed/swat/train.csv'),
        ('data/original/swat/SWaT_Dataset_Attack_v0.xlsx', 'data/processed/swat/test.csv')
    ]
    for args in swat_list:
        preprocess_swat(*args)
