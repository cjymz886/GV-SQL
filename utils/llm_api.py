"""LLM API"""

import openai

def openai_llm_api(prompt=None,msg=None,max_tokens=2048,stop=None,):
	client = openai.OpenAI(
		base_url = "http://10.31.10.163:10004/",
		api_key = "sk-pOwqtW732-7E7uCLVVbOhQ"  

	)
	if prompt and not msg:
		messages = [{"role": "user", "content":prompt}]
	elif not prompt and msg:
		messages = msg
	else:
		raise ValueError("Input error")

	try:
		response = client.chat.completions.create(
			model = "MiniMax-M3-Internal",  #DeepSeek-V4-Flash-Internal,MiniMax-M3-Internal
			messages = messages,
			temperature=1.0,
			max_tokens=max_tokens,
			top_p=0.95,
			stream=False,
			stop=stop,
			extra_body={"chat_template_kwargs": {
			"enable_thinking": True,
				}
			},
		)
		response = response.choices[0].message.content
		return response
	except Exception as e:
		return None
	
