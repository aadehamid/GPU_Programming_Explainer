import modal

# Reference the existing image by ID
image = modal.Image.from_id("im-NjblKWrA2QbdjQgx0s2zRG")

app = modal.App("temp-shell", image=image)

@app.function()
def shell():
    pass
