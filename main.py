import openai
from supabase import create_client, Client
import tiktoken
import numpy as np
from openai.embeddings_utils import  cosine_similarity
import requests
from keys import OPENAIAPIKEY,SUPABASEKEY,SUPABASEURL


openApiKey = OPENAIAPIKEY
openai.api_key = OPENAIAPIKEY
supabaseURL = SUPABASEURL
supabaseKey = SUPABASEKEY
supabaseClient: Client = create_client(supabaseURL, supabaseKey)




# cl100k_base
def numTokensFromString(string: str, encoding_name: str) -> int:
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens


def startDeployment(name_of_food: str, context: str, token_count: int, embedding):
    print('-' * 10 + 'Deployment One-Item Supabase-Database Start' + '-' * 10)
    supabaseClient.table('items').insert(
        {'name_of_food': name_of_food,
         'context': context,
         'token_count': token_count,
         'embedding': embedding}
    ).execute()
    print('-' * 10 + 'Deployment Finish' + '-' * 10)


def createDataEmbedding(context: str):
    print('-' * 10 + 'Embedding Creating...' + '-' * 10)
    response = openai.Embedding.create(
        input=context,
        model="text-embedding-ada-002"
    )
    print('-' * 10 + 'Embedding Created' + '-' * 10)
    return response['data'][0]['embedding']


# def cosine_similarity(a, b):
#     nominator = np.dot(a, b)
#
#     a_norm = np.sqrt(np.sum(a ** 2))
#     b_norm = np.sqrt(np.sum(b ** 2))
#
#     denominator = a_norm * b_norm
#
#     cosine_similarity = nominator / denominator
#
#     return cosine_similarity


def readData(filename: str):
    data = open(filename, 'r').readlines()
    data = [i.strip('\n') for i in data]
    data = ''.join(data).split('****')
    data = list(filter(lambda a: a != '', data))
    # print(data)
    return data


#
# # Deployment to the DataBase
# data = readData('data.txt')
# for x in data:
#     token_number = numTokensFromString(x,'cl100k_base')
#     if token_number >= 1000: pass
#     else:
#         embedding = createDataEmbedding(x)
#         startDeployment(name_of_food='Yemek',context=x,token_count=token_number,embedding=embedding)
#
#
def completionOpenAI(object:dict):
    url = 'https://api.openai.com/v1/completions'
    response = requests.post(url,headers={
        'Authorization': f'Bearer {openApiKey}',
        'Content-Type': 'application/json',
      }, data=object)


# User Interface
def userInterface():
    match_thresold = 0.85
    print(20 * '+')
    request = input('Merhaba ben Tarif-GPT, bilmek istediğini bana sorabilirsin ? ')
    print(20 * '+')
    embeddingdata = createDataEmbedding(request.strip().replace('\n', ''))
    #Postgre Function does not work.
    # rpc(match_items,{embedding,match_thresold : 0.78,match_count:10,min_content_length :50})
    # data = supabaseClient.rpc('match_items',
    #                           {'embedding': embeddingdata, 'match_threshold': 0.0002, 'match_count': 100,
    #                            'min_content_length': 1}).execute()
    converted_data_request = np.array(embeddingdata).astype('float64')
    data_embedding = supabaseClient.table('items').select('*').execute()
    context = ''
    print(20 * '+')
    print("I'm thinking")
    for m in range(len(data_embedding.data)):
        a = np.array(eval(data_embedding.data[m]['embedding'])).astype('float64')
        x = cosine_similarity(a, converted_data_request)
        if x > match_thresold:
            context = context + str(data_embedding.data[m]['context'])
    print(20 * '+')
    prompt = f'''
        You are a Recipe-GPT, you have a data of food that includes Name,Ingredients,CALORI."
          Context sections:
          {context}

          Question: """
          {request}
          """
          Answer as markdown (including related code snippets if available):
          '''
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {'role':'user','content':prompt}
        ]
    )
    print(response['choices'][0]['message']['content'])


userInterface()
