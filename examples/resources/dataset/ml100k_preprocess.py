import pandas as pd
class Ml100k():

    def __init__(self,dir):
        #self.data_dir = data_dir
        self.dir=dir
        self.data = self.load_data()

    

    def load_data(self):
        data = pd.read_csv(f'{self.dir}/ml100k/raw/u.data', sep='\t', names=['user_id', 'item_id', 'rating', 'timestamp'])
        data = data.drop(columns='timestamp')
        return data
    

    def get_item_info(self):
        movie_info=pd.read_csv(f"{self.dir}/ml100k/raw/u.item", sep='|', names=['item_id', 'title', 'release_date', 'video_release_date', 'IMDb_URL', 'unknown', 'Action', 'Adventure', 'Animation', 'Children', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western'], encoding='latin-1')
        return movie_info
    
    def get_user_info(self):
        user_info=pd.read_csv(f"{self.dir}/ml100k/raw/u.user", sep='|', names=['user_id', 'age', 'occupation', 'zip_code'], encoding='latin-1')
        return user_info
    

    def merge_data(self):
        item_info = self.get_item_info()
        user_info = self.get_user_info()
        data=self.load_data()
        data = pd.merge(data, item_info, on='item_id', how='left')
        data = pd.merge(data, user_info, on='user_id',how='left')
        data.drop(columns=['IMDb_URL', 'video_release_date', 'release_date', 'zip_code','title'], inplace=True)
        data['user_id']=data['user_id']-1
        data['item_id']=data['item_id']-1


        # convert rating to 1 if rating > 3 else 0
        data['rating'] = data['rating'].apply(lambda x: 1 if x>=3 else 0)
        return data
    
    def get_data(self): 
        to_return=self.merge_data()
        return to_return
