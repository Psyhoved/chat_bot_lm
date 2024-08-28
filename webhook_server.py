from fastapi import FastAPI, Request
import os
import subprocess

app = FastAPI()
CONTAINER_NAME = 'chat_bot_august'
EX_CONTAINER_NAME = 'chat_bot_july'
EX_IMAGE_NAME = 'chat_bot_july'


def container_exists(container_name):
    result = subprocess.run(['docker', 'ps', '-a', '-q', '-f', f'name={container_name}'], stdout=subprocess.PIPE)
    return result.stdout.strip() != b''


def image_exists(image_name):
    result = subprocess.run(['docker', 'images', '-q', image_name], stdout=subprocess.PIPE)
    return result.stdout.strip() != b''


@app.post("/webhook-endpoint")
async def webhook(request: Request):
    # Команда для обновления кода и пересборки Docker-контейнера
    os.chdir('/root/chat_bot_lm')

    # Выполнение команд
    subprocess.run(['git', 'pull'])
    subprocess.run(['docker', 'build', '-t', f'{CONTAINER_NAME}_image', '.'])

    # Удаляем образа с прошлого релиза, если он существует
    if image_exists(EX_IMAGE_NAME):
        subprocess.run(['docker', 'rmi', EX_IMAGE_NAME])

    # Удаляем контейнер с прошлого релиза, если он существует
    if container_exists(EX_CONTAINER_NAME):
        # Остановка и удаление контейнера, если он существует
        subprocess.run(['docker', 'stop', EX_CONTAINER_NAME])
        subprocess.run(['docker', 'rm', EX_CONTAINER_NAME])

    # Проверка, существует ли контейнер
    if container_exists(CONTAINER_NAME):
        # Остановка и удаление контейнера, если он существует
        subprocess.run(['docker', 'stop', CONTAINER_NAME])
        subprocess.run(['docker', 'rm', CONTAINER_NAME])

    subprocess.run(['docker', 'run', '-d', '--name', CONTAINER_NAME, '-p', '8000:8000', f'{CONTAINER_NAME}_image'])

    return {"message": "Success"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=80, reload=True)
