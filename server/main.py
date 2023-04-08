from flask import Flask, request, jsonify, render_template
import util
import util2
import base64
import json
app=Flask(__name__)
# @app.route("/")
# def main():
#     return render_template("home.html")

@app.route("/classify",methods=["GET","POST"])
def classify_image():

    # if request.method == "POST":
    # print( request.files)
    image=request.files['image']
    img_string=base64.b64encode(image.read())
    print(img_string)
    head=util.image_classify(img_string)
    # resp2 = util2.getNews(head[0]["celeb"])

    # print()
    response=jsonify(head)
    response.headers.add("Access-Control-Allow-Origin","*")


    return response


@app.route("/pagemaster",methods=["GET","POST"])
def pagefiller():
    name=request.args["name"]
    per=float(request.args["p"])
    resp2 = util2.getNews(name)
    handle=None
    with open("./artefacts/celeb_handles.json", "r") as f:
        handle = json.load(f)
    print(resp2)

    return render_template("load_page.html",name=name,per=per,news=resp2,handles=handle[name]);



if __name__=="__main__":
    app.run(port=3000,debug=True)