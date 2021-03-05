FROM python:3.7
RUN pip install pipenv

COPY . /app
WORKDIR /app

RUN pip install pipenv
RUN pipenv install --system --deploy

CMD ["python", "inference.py", "--url", "https://image.made-in-china.com/202f0j00yUrfpcYJYekG/Propeller-Hat-Colorful-Patchwork-Custom-Design-Cotton-Funny-Baseball-Hats.jpg"]