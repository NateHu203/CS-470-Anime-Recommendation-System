import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, MultiLabelBinarizer
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# 设置显示选项，提高可读性
pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)

print("加载数据集...")
# 读取数据
rating_df = pd.read_csv('dat/rating.csv')
anime_df = pd.read_csv('dat/anime.csv')

# 检查重复条目
print(f"重复的用户-动漫对: {rating_df.duplicated(subset=['user_id', 'anime_id']).sum()}")
print(f"重复的动漫条目: {anime_df.duplicated(subset=['anime_id']).sum()}")

# 删除重复项
rating_df = rating_df.drop_duplicates(subset=['user_id', 'anime_id'])

# 处理缺失值和异常值
print("处理缺失值和异常值...")

# 将-1标记为缺失值（用户观看但未评分）
rating_df['rating'] = rating_df['rating'].replace(-1, np.nan)

# 处理动漫数据集中的缺失值
anime_df['genre'] = anime_df['genre'].fillna('')
anime_df['type'] = anime_df['type'].fillna('Unknown')

# 将字符串列转换为数值类型
try:
    # 尝试将episodes转换为数值类型
    anime_df['episodes'] = pd.to_numeric(anime_df['episodes'], errors='coerce')
    # 使用中位数填充缺失值
    anime_df['episodes'] = anime_df['episodes'].fillna(anime_df['episodes'].median())
except:
    # 如果转换失败，则用0填充
    anime_df['episodes'] = anime_df['episodes'].fillna(0)

# 确保评分列是数值类型
anime_df['rating'] = pd.to_numeric(anime_df['rating'], errors='coerce')
anime_df['rating'] = anime_df['rating'].fillna(anime_df['rating'].median())  

# 确保members列是数值类型
anime_df['members'] = pd.to_numeric(anime_df['members'], errors='coerce')
anime_df['members'] = anime_df['members'].fillna(0)

# 特征工程
print("执行特征工程...")

# 1. 处理分类特征
#dummy variables
type_df = pd.get_dummies(anime_df['type'], prefix='type')

# 使用MultiLabelBinarizer处理类别
anime_df['genre_list'] = anime_df['genre'].str.split(', ')
mlb = MultiLabelBinarizer()
genre_matrix = mlb.fit_transform(anime_df['genre_list'])
genre_df = pd.DataFrame(genre_matrix, columns=mlb.classes_)

# 2. 添加派生特征
# 计算流行度指标
anime_count = rating_df.groupby('anime_id').size().reset_index(name='rating_count')
anime_avg = rating_df.groupby('anime_id')['rating'].mean().reset_index(name='user_avg_rating')
anime_median = rating_df.groupby('anime_id')['rating'].median().reset_index(name='user_median_rating')
anime_std = rating_df.groupby('anime_id')['rating'].std().reset_index(name='user_rating_std')

# 计算流行度百分位数（对于冷启动推荐有用）
anime_count['popularity_percentile'] = anime_count['rating_count'].rank(pct=True) * 100

# 将流行度指标加入动漫数据集
anime_df = anime_df.merge(anime_count, on='anime_id', how='left')
anime_df = anime_df.merge(anime_avg, on='anime_id', how='left')
anime_df = anime_df.merge(anime_median, on='anime_id', how='left')
anime_df = anime_df.merge(anime_std, on='anime_id', how='left')

# 填充NaN值
anime_df['rating_count'] = anime_df['rating_count'].fillna(0)
anime_df['user_avg_rating'] = anime_df['user_avg_rating'].fillna(0)
anime_df['user_median_rating'] = anime_df['user_median_rating'].fillna(0)
anime_df['user_rating_std'] = anime_df['user_rating_std'].fillna(0)
anime_df['popularity_percentile'] = anime_df['popularity_percentile'].fillna(0)

# 3. 用户行为特征
user_stats = rating_df.groupby('user_id').agg(
    user_mean_rating=('rating', 'mean'),
    user_median_rating=('rating', 'median'),
    user_rating_count=('rating', 'count'),
    user_rating_std=('rating', 'std')
).reset_index()

# 计算用户参与度百分位数
user_stats['engagement_percentile'] = user_stats['user_rating_count'].rank(pct=True) * 100

# 为user_rating_std填充NaN值（仅有一个评分的用户）
user_stats['user_rating_std'] = user_stats['user_rating_std'].fillna(0)

# 将用户统计数据加入评分数据集
rating_df = rating_df.merge(user_stats, on='user_id', how='left')

# 4. 添加归一化评分以考虑用户偏差（zscore）
rating_df['normalized_rating'] = rating_df.apply(
    lambda x: (x['rating'] - x['user_mean_rating']) / max(x['user_rating_std'], 0.001) if pd.notnull(x['rating']) else np.nan, 
    axis=1
)

# 5. 创建基于内容的特征（最常见的流派和类型）
user_genres = []
for user_id in rating_df['user_id'].unique():
    user_anime_ids = rating_df[rating_df['user_id'] == user_id]['anime_id'].tolist()
    user_anime = anime_df[anime_df['anime_id'].isin(user_anime_ids)]
    
    # 扁平化流派列表并计算出现次数
    all_genres = []
    for genres in user_anime['genre_list'].dropna():
        if isinstance(genres, list) and genres:
            all_genres.extend(genres)
    
    genre_counts = pd.Series(all_genres).value_counts().to_dict() if all_genres else {}
    
    user_genres.append({
        'user_id': user_id,
        'top_genres': str([genre for genre, count in genre_counts.items()][:3]) if genre_counts else "[]",
        'genre_diversity': len(genre_counts)
    })

user_genre_df = pd.DataFrame(user_genres)
rating_df = rating_df.merge(user_genre_df, on='user_id', how='left')

# 合并动漫特征
anime_features = pd.concat([
    anime_df[['anime_id', 'name', 'episodes', 'rating', 'members', 
              'rating_count', 'user_avg_rating', 'user_median_rating', 
              'user_rating_std', 'popularity_percentile']],
    genre_df,
    type_df
], axis=1)

# 6. 归一化所有数值特征以防止偏差
numerical_features = ['episodes', 'members', 'rating_count', 'user_avg_rating', 
                      'user_median_rating', 'user_rating_std', 'popularity_percentile']

# 检查并确保所有要归一化的列都是数值类型
for feature in numerical_features:
    if feature in anime_features.columns:
        # 确保列是数值类型
        anime_features[feature] = pd.to_numeric(anime_features[feature], errors='coerce')
        # 填充缺失值
        anime_features[feature] = anime_features[feature].fillna(0)

# 使用MinMaxScaler进行有界特征（使值保持在0和1之间）
# 只对非零值的列进行归一化
valid_numerical_features = [col for col in numerical_features 
                            if col in anime_features.columns and anime_features[col].sum() > 0]

if valid_numerical_features:
    mm_scaler = MinMaxScaler()
    anime_features[valid_numerical_features] = mm_scaler.fit_transform(anime_features[valid_numerical_features])

# 验证特征创建
print(f"动漫特征数据集形状: {anime_features.shape}")
print(f"增强评分数据集形状: {rating_df.shape}")

# 拆分数据
print("将数据拆分为训练集和测试集...")

# 选项1：简单随机拆分
train_rating, test_rating = train_test_split(
    rating_df.dropna(subset=['rating']),
    test_size=0.2,
    random_state=42
)

# 选项2：基于用户的拆分，模拟真实场景
# 确保测试集中的每个用户在训练集中都有一些历史记录
unique_users = rating_df['user_id'].unique()
train_users, test_users = train_test_split(unique_users, test_size=0.2, random_state=42)

# 对于测试集中的每个用户，将他们的评分拆分为80/20
train_indices = []
test_indices = []

for user_id in unique_users:
    user_ratings = rating_df[rating_df['user_id'] == user_id].dropna(subset=['rating'])
    
    if len(user_ratings) >= 5:  # 用户有足够的评分
        if user_id in test_users:
            # 这是一个测试用户 - 将他们80%的评分放入训练集，20%放入测试集
            user_train, user_test = train_test_split(
                user_ratings, 
                test_size=0.2,
                random_state=42
            )
            train_indices.extend(user_train.index)
            test_indices.extend(user_test.index)
        else:
            # 这是一个仅训练的用户
            train_indices.extend(user_ratings.index)
    else:
        # 用户没有足够的评分 - 全部放入训练集
        train_indices.extend(user_ratings.index)

# 创建拆分数据集
train_rating_v2 = rating_df.loc[train_indices]
test_rating_v2 = rating_df.loc[test_indices]

print(f"训练集大小（选项1）: {len(train_rating)}")
print(f"测试集大小（选项1）: {len(test_rating)}")
print(f"训练集大小（选项2）: {len(train_rating_v2)}")
print(f"测试集大小（选项2）: {len(test_rating_v2)}")

# 保存预处理数据
print("将预处理数据保存到文件...")
anime_features.to_csv('dat/processed/anime_features_normalized.csv', index=False)
rating_df.to_csv('dat/processed/rating_enhanced.csv', index=False)
# train_rating.to_csv('processed/dat/rating_train.csv', index=False)
# test_rating.to_csv('processed/dat/rating_test.csv', index=False)
train_rating_v2.to_csv('dat/processed/rating_train_v2.csv', index=False)
test_rating_v2.to_csv('dat/processed/rating_test_v2.csv', index=False)

print("预处理完成!")