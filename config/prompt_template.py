schema_enhanced_prompt="""You are a data analyst tasked with generating insightful description for the column of a table that conform to database specifications. Your goal is to provide clear and concise explanations that improve the understandability of the data.

# About Column description
- step1: interpretation of a column name is contingent upon its relational association with the table name. 
- step2: the generated description should explain the column meaning within the context of table_name
- setp3: output overall column description according to the Examples

# About Important notes
- The input is Input, and you can make use of all of this information.
- Please keep the input column names unchanged, in other words, just repeat.
- The generated description should be brief and not too long.

# Examples
input: database_id='debit_card_specializing',table_name='customers',column_name='CustomerID',column_type='integer',column_description='identification of the customer',example_values=[49352, 38791, 17077, 49292, 48507]
output: CustomerID is an integer column in the customers table, which is used to identify the customer.

input: database_id='debit_card_specializing',table_name='customers',column_name='client_segment',column_type='text',column_alias_name='client segment',all_possible_values=['SME', 'LAM', 'KAM']
output: This column represents the client segment of the customers table in the debit_card_specializing database. The client segment is a text column that can take on one of the following values: SME, KAM, LAM.

# Input
{input_paras}

Begin! Take a deep breath and think logically. Output result Only."""







identify_column_in_SQL_prompt="""You are a professional SQL expert. Given a SQL query, identify the table names involved and their corresponding column names.

# Rules
- For table names, consider FROM clauses as well as multi-table queries involving JOIN, UNION, EXISTS/IN, etc.
- For column names, consider WHERE clauses as well as ORDER BY and other such forms.
- Ensure that tables and column names are correctly aligned, not confused, and deduplicated.
- Treat * as a column name representing all columns.
- Keep the identified table names and column names exactly consistent with the original SQL; do not change their case.

# Examples
SQL: SELECT AVG(movie_popularity) FROM movies WHERE director_name = 'Stanley Kubrick'
output: {{'movies': ['movie_popularity','director_name']}}

SQL: SELECT AVG(T2.rating_score) FROM movies AS T1 INNER JOIN ratings AS T2 ON T1.movie_id = T2.movie_id WHERE T1.movie_title = 'When Will I Be Loved'
output: {{'movies': ['movie_id','movie_title'],'ratings':['movie_id','rating_score']}}

SQL: SELECT T3.user_avatar_image_url, T3.rating_date_utc FROM movies AS T1 INNER JOIN ratings AS T2 ON T1.movie_id = T2.movie_id INNER JOIN ratings_users AS T3 ON T3.user_id = T2.user_id WHERE T3.user_id = 41579158 ORDER BY T3.rating_date_utc DESC LIMIT 1
output: {{'movies': ['movie_id'],'ratings':['movie_id','user_id'],'ratings_users':['user_avatar_image_url','rating_date_utc','user_id']}}

SQL: SELECT CAST(SUM(CASE WHEN user_subscriber = 1 THEN 1 ELSE 0 END) AS REAL) * 100 / COUNT(*) FROM ratings
output: {{'ratings': ['*']}}

SQL: SELECT COUNT(*) FROM ratings WHERE movie_id = 1269 AND rating_score <= 2 AND user_eligible_for_trial = 1 AND user_has_payment_method = 1
output: {{'ratings': ['*','movie_id','rating_score','user_eligible_for_trial','user_has_payment_method']}}

SQL: SELECT `Free Meal Count (K-12)` / `Enrollment (K-12)` FROM frpm WHERE `County Name` = 'Alameda' ORDER BY (CAST(`Free Meal Count (K-12)` AS REAL) / `Enrollment (K-12)`) DESC LIMIT 1
output: {{'frpm': ['Free Meal Count (K-12)','County Name','Enrollment (K-12)']}}

The input SQL is: {sql_input}. Please refer to the Examples and output the corresponding result in JSON format. Do not provide any explanation; output directly.
"""








question_keywords_extract_prompt="""You are an experienced database expert. Please extract Keywords from the Question and Evidence provided below.

# About Important notes
- The extracted Keywords are those that may be related to tables or columns.
- Keywords can have different semantic granularities，such as 'movie' and 'popular movie'.
- Refer to the Examples below and output extracted keywords with '|'. Besides, remove duplicates keywords.


# Examples
Question: State the most popular movie? When was it released and who is the director for the movie?
Evidence: most popular movie refers to MAX(movie_popularity); when it was released refers to movie_release_year; director for the movie refers to director_name;
Keywords: movie|director|most popular movie|movie_popularity|movie_release_year|director_name

Question: Among all the users who received the high number of compliments, what percent received the 'cute' type of compliment.
Evidence: high number of compliments refers to number_of_compliments = 'High'; percentage = divide(count(user_id where compliment_type = 'cute'), count(user_id))*100%
Keywords: users|high number of compliments|compliments|number_of_compliments|percent|percentage|compliment_type|user_id

# Input
{input_paras}

Output Keywords of Input directly, without any explanation.
"""




sql_think_output_extract_prompt="""You are an experienced database expert . Please based on Database schema , the question and the evidence , write a thought process for generating the SQL query.

##Database schema
{schema}

## Instructions:
- The thinking process should demonstrate how to solve problems step by step , as well as how to utilize schema and evidence information.
- The SQL is the ground-truth answer and is not available during actual reasoning. Therefore, your thought process must not reference or assume knowledge of the SQL. Instead, it should describe how to gradually derive or approximate the gold SQL, using only the provided information as the sole basis for each reasoning step.
- The thinking process should not be too lengthy and the word count should be kept within 512 .

## Input
question: {question}
evidence: {evidence}
SQL: {sql}

Take a deep breath and think step by step to output thought process . 
"""







sql_think_output_extract_with_vl_prompt="""You are an experienced database expert . Please based on images, Database schema , the question and the evidence , write a thought process for generating the SQL query.

##Database schema
{schema}

## Instructions:
- The thinking process should demonstrate how to solve problems step by step , as well as how to utilize E-R image , schema and evidence information.
- The SQL is the ground-truth answer and is not available during actual reasoning. Therefore, your thought process must not reference or assume knowledge of the gold SQL. Instead, it should describe how to gradually derive or approximate the correct SQL, using only the provided information as the sole basis for each reasoning step.
- The thinking process should not be too lengthy and the word count should be kept within 521 .

## Input
question: {question}
evidence: {evidence}
SQL: {sql}

Take a deep breath and think step by step to output thought process . 
"""




sql_generation_prompt="""Task Overview : You are a data science expert . Below , you are provided with a database schema and a natural language question . Your task is to understand the schema and generate a valid SQL query to answer the question .

## Database Engine:
{Dialect}

## SQL Reserved words
clause keywords: SELECT , FROM , WHERE , GROUP , BY , ORDER , UNION , LIMIT , HAVING , DISTINCT , AS , BETWEEN , LIKE , ALL , ON , PARTITION , SUBSTRING , INTEGER , USING
join keywords: INNER JOIN , LEFT JOIN , RIGHT JOIN , JOIN , ON , CROSS JOIN , LEFT OUTER JOIN
where operations: IS , NOT , NULL , NONE , IN , = , > , < , >= , <= , != , <> , OF , TRUE , NO , FALSE
sql operations: AVG, COUNT, MAX , MIN , ROUND , ABS , SUM , LENGTH , CAST , SUBSTR , INSTR , REPLACE
cond operations: IIF , AND , OR , CASE , END , WHEN , THEN , ELSE , IF , EXISTS
date operations: NOW , CURDATE , STRFTIME , TIMESTAMPDIFF
unit operations: - , + , * , /
oder operations: DESC , ASC

## Database Schema:
# Database: {Database}
{Schema}
This markdown schema describe the database's structure , including tables , columns , primary keys , foreign keys , and any relevant relationships or constraints .

## Examples:
{Examples}
Provide several examples similar to the question , along with their corresponding SQL , for your reference in SQL generation .

## Question:
question: {question}
evidence: {evidence}

## Instructions:
- If Matched contents is provided , you can use it as reference when generating the SQL query .
- Make sure you only output the information that is asked in the question. If the question asks for a specific column, make sure to only include that column in the SELECT clause , nothing more .
- The generated query should return all of the information asked in the question without any missing or extra information .
- Make sure table name and column name in generated SQL query are consistent with the schema defined above .
- Make sure sql keywords in generated SQL query are consistent with the SQL Reserved words defined above.


## Output Format:
In your answer , please enclose the generated SQL query in a code block:
```
-- Your SQL query
```

Take a deep breath and think step by step to find the correct SQL query.
"""



sql_generation_with_vl_prompt="""{input_image}/nTask Overview :You are a data science expert . Below, you are provided with images and schema of a database, and a natural language question . Your task is to understand the schema and generate a valid SQL query to answer the question .

## Database Engine:
{Dialect}

## SQL Reserved words
clause keywords: SELECT , FROM , WHERE , GROUP , BY , ORDER , UNION , LIMIT , HAVING , DISTINCT , AS , BETWEEN , LIKE , ALL , ON , PARTITION , SUBSTRING , INTEGER , USING
join keywords: INNER JOIN , LEFT JOIN , RIGHT JOIN , JOIN , ON , CROSS JOIN , LEFT OUTER JOIN
where operations: IS , NOT , NULL , NONE , IN , = , > , < , >= , <= , != , <> , OF , TRUE , NO , FALSE
sql operations: AVG, COUNT, MAX , MIN , ROUND , ABS , SUM , LENGTH , CAST , SUBSTR , INSTR , REPLACE
cond operations: IIF , AND , OR , CASE , END , WHEN , THEN , ELSE , IF , EXISTS
date operations: NOW , CURDATE , STRFTIME , TIMESTAMPDIFF
unit operations: - , + , * , /
oder operations: DESC , ASC

## Database Schema:
# Database: {Database}
{Schema}
The input image is a ddl schema and The Database Schema is a markdown schema , which both including tables , columns , primary keys , foreign keys , and any relevant relationships or constraints .

## Examples:
{Examples}
Provide several examples similar to the question, along with their corresponding SQL, for your reference in SQL generation .

## Instructions:
- If Matched contents is provided , you can use it as reference when generating the SQL query.
- Make sure you only output the information that is asked in the question. If the question asks for a specific column, make sure to only include that column in the SELECT clause, nothing more.
- The generated query should return all of the information asked in the question without any missing or extra information.
- Make sure table name and column name in generated SQL query are consistent with the ddl schema defined above.
- Make sure sql keywords in generated SQL query are consistent with the SQL Reserved words defined above.

## Question:
question: {question}
evidence: {evidence}

## Output Format:
In your answer, please enclose the generated SQL query in a code block:
```
-- Your SQL query
```

Take a deep breath and think step by step to find the correct SQL query.
"""





sql_refinement_think_output_extract_with_vl_prompt="""You are a data science expert. Below , you are provided with images , ##Database schema , ## Question (including question,evidence,gold_SQL ), Incorrect SQL (including predict_SQL query and Error clue) . Given these relevant information , your task is to generate a thought process that explains how to repair the incorrect predict_SQL step by step. The thought process should explicitly reference the provided information to justify each correction .

##Database schema
{schema}

## Question
question: {question}
evidence: {evidence}
gold_SQL: {gold_sql}

## Incorrect SQL
predict_SQL: {pred_sql}
Error clue: {ERROR}

## Instructions:
- The thinking process should demonstrate how to solve problem step by step , as well as how to utilize images , schema ,evidence and Error clue information.
- The gold_SQL is the ground-truth answer and is not available during actual reasoning. Therefore, your thought process must not reference or assume knowledge of the gold_SQL. Instead, it should describe how to gradually derive or approximate the correct SQL, using only the provided information as the sole basis for each reasoning step.
- The thinking process should not be too lengthy and the word count should be kept within 512 .

Take a deep breath and think step by step to output thought process . 
"""



sql_refinement_prompt="""You are a data science expert . Below , you are provided with a database schema , a natural language question , and a incorrect SQL query. Your task is to understand the schema and repair the incorrect SQL to answer the question .

## Database Engine:
{Dialect}

## Database Schema:
# Database: {Database}
{Schema}
This markdown schema describe the database's structure , including tables , columns , primary keys , foreign keys , and any relevant relationships or constraints .

## Question:
question: {question}
evidence: {evidence}

## Incorrect SQL
SQL: {SQL}
Error clue: {ERROR}

## Instructions:
- Carefully analyze the reasons for the error clue and find a reasonable solution to fix the incorrect SQL .
- If it is a grammar issue, adjust the structure of the SQL query to make it more reasonable .
- If the database information is incorrect, please fix this error based on the provided schema information.
- If the result of the execution is incorrect, please re-understand the requirements of the question and generate a more reasonable query again.

## Output Format:
In your answer , please enclose the generated SQL query in a code block:
```
-- Your SQL query
```

Take a deep breath and think step by step to find the correct SQL query.
"""


sql_refinement_with_vl_prompt="""{input_image}/nYou are a data science expert . Below , you are provided with images , a database schema , a natural language question , and a incorrect SQL query. Your task is to understand the schema and repair the incorrect SQL to answer the question .

## Database Engine:
{Dialect}

## Database Schema:
# Database: {Database}
{Schema}
The input images and this markdown schema describe the database's structure , including tables , columns , primary keys , foreign keys , and any relevant relationships or constraints .

## Question:
question: {question}
evidence: {evidence}

## Incorrect SQL
SQL: {SQL}
Error clue: {ERROR}

## Instructions:
- Carefully analyze the reasons for the error clue and find a reasonable solution to fix the incorrect SQL .
- If it is a grammar issue, adjust the structure of the SQL query to make it more reasonable .
- If the database information is incorrect, please fix this error based on the provided schema information.
- If the result of the execution is incorrect, please re-understand the requirements of the question and generate a more reasonable query again.

## Output Format:
In your answer , please enclose the generated SQL query in a code block:
```
-- Your SQL query
```

Take a deep breath and think step by step to find the correct SQL query."""




