from server import app

if __name__ == '__main__':
    app.run(debug=True, host="localhost", port=5000, use_reloader=False)
