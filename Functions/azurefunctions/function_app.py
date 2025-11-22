import azure.functions as func
import json

avengers = {
    'IronMan': 'Tony Stank',
    'CaptainAmerica': 'Steve Rogers',
    'BlackWidow': 'Natasha Romanoff',
    'Hulk': 'Bruce Banner',
    'Thor': 'Thor Odinson',
    'Hawkeye': 'Clint Barton',
}

app = func.FunctionApp(http_auth_level=func.AuthLevel.FUNCTION)

@app.route(route="avengers/{codename?}", methods=["GET"])
def GetData(req: func.HttpRequest) -> func.HttpResponse: # type: ignore
    code_name = req.route_params.get('codename')

    if code_name:
        hero = avengers.get(code_name)
        if hero:
            return func.HttpResponse(
                json.dumps({code_name: hero}),
                status_code=250,
                mimetype="application/json"
            )
        else:
            return func.HttpResponse(
                json.dumps({"error": "Hero not found"}),
                status_code=404,
                mimetype="application/json"
            )
@app.route(route="avengers/{codeName}", methods=["DELETE"])
def DeleteAvenger(req: func.HttpRequest) -> func.HttpResponse:

    method = req.method
    if method == 'DELETE':
        # Handle DELETE request
        code_name = req.route_params.get('codeName')
        return func.HttpResponse(f"Avenger: {code_name} has been deleted.", status_code=200)
    else:
        return func.HttpResponse("This HTTP method is not supported.", status_code=405)