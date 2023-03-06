import os
from fastapi import FastAPI, BackgroundTasks, UploadFile, Form, File, Query
import uvicorn
from search import Inference
from config import CFG
app = FastAPI()

inference = Inference(CFG)


@app.get("/search")
async def search(question):

    final_results = inference.search(question, top_k=20)
    #print( final_results)
    answer, title = inference.answering(final_results)

    return answer




if __name__ == '__main__':
    uvicorn.run(app, 
                host="0.0.0.0", 
                port=8080, 
                log_level="info", 
                # reload=False,
                workers=1)