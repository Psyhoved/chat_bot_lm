# chat_bot_lm ![Coverage](./coverage.svg)
## Приложение с чатботом поддержки сети Жизньмарт, отвечающим пользователю на основе базы знаний ЖМ
Для работы необходима база знаний в формате pdf в корне проекта (на сервере обновлять вручную)

Используется сплитер базы знаний на чанки по символу ~

Основы в [блокноте](https://colab.research.google.com/drive/1yLOW4CT_CCsrBUzIbs74uT4avTwvKxA5?usp=sharing)

## Команды для запуска докера на сервере

### 1. Остановка и удаление прошлых контейнеров и образов (следить за переполнением памяти в докере)

sudo docker stop old_container_name

sudo docker container prune -f

sudo docker image prune -a -f

### 2. Сборка нового образа 

sudo docker build -t image_name .

### 4. Запуск контейнера с новым образом

sudo docker run -d --name container_name -p 8000:8000 image_name

### 5. Просмотр логов

sudo docker logs container_name


### По всем вопросам можно обращаться к Саше (https://t.me/HardWorkPlayer_2_0)