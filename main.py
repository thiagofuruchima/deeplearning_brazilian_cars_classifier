from app import create_app

application = create_app()

if __name__ == '__main__':
    # This is used when running locally. Gunicorn is used to run the
    # application on Google App Engine. See entrypoint in app.yaml.
    # Using "application" instead of the standard "app" to prevent errors.
    application.run(host='0.0.0.0', port=3002, debug=True)