from fastapi import FastAPI,HTTPException

app = FastAPI()

all_post = {"1": {"title": "First Post", "content": "This is the content of the first post."},
            "2": {"title": "Second Post", "content": "This is the content of the second post."}}


@app.get("/posts")
def get_all_posts(limit:int = 0):
    if limit:
        return dict(list(all_post.items())[:limit])
    return all_post

@app.get("/posts/{post_id}")
def get_post(post_id: str):
    if post_id not in all_post:
        raise HTTPException(status_code=404, detail="Post not found")
    return all_post.get(post_id, {"error": "Post not found"})

