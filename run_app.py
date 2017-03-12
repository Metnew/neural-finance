from server import app
import optparse

if __name__ == '__main__':
    parser = optparse.OptionParser()

    parser.add_option("-P", "--port",
                      help="Port for the Flask app " + \
                           "[default %s]" % 5000,
                      default=5000)

    options, _ = parser.parse_args()


    app.run(debug=True, host="localhost", port=int(options.port), use_reloader=True)
