ё#!/usr/bin/env bash
# Разворачивает стек bot-detector локально: MLflow, сервис, Prometheus, Grafana.
#
#   ./deploy.sh            # собрать и поднять; обучить модель, если в реестре ещё нет champion
#   ./deploy.sh --train    # то же, но обучить новую версию в любом случае
#
# Скрипт можно запускать повторно: compose пересоздаёт только изменившиеся сервисы.
set -euo pipefail

cd "$(dirname "$0")"

MLFLOW_URL=http://localhost:5050
API_URL=http://localhost:8000
MODEL_NAME=bot-detector
MODEL_ALIAS=champion

FORCE_TRAIN=false
case "${1:-}" in
  "") ;;
  --train) FORCE_TRAIN=true ;;
  -h|--help) sed -n '2,7p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
  *) echo "Неизвестный аргумент: $1 (см. --help)" >&2; exit 1 ;;
esac

log() { printf '\n==> %s\n' "$*"; }
die() { printf 'Ошибка: %s\n' "$*" >&2; exit 1; }

# wait_for <название> <url> <секунд>: ждёт, пока url ответит 2xx
wait_for() {
  local name=$1 url=$2 timeout=$3 waited=0
  until curl -fs -o /dev/null "$url"; do
    (( waited >= timeout )) && die "$name не ответил за ${timeout} с (docker compose logs)"
    sleep 3
    waited=$((waited + 3))
  done
  echo "$name готов"
}

log "Проверка окружения"
command -v docker >/dev/null || die "не найден docker"
docker info >/dev/null 2>&1 || die "Docker не запущен"
docker compose version >/dev/null 2>&1 || die "нет плагина docker compose"
for f in train.json test.json ytrain.csv ytest.csv; do
  [[ -f data/$f ]] || die "нет файла data/$f — он нужен для обучения"
done
echo "ok"

log "Сборка образа и запуск сервисов"
docker compose up -d --build

log "Ожидание MLflow"
wait_for MLflow "$MLFLOW_URL/health" 120

# 200 — алиас есть, 404 — модель ещё не обучалась
alias_status=$(curl -s -o /dev/null -w '%{http_code}' \
  "$MLFLOW_URL/api/2.0/mlflow/registered-models/alias?name=$MODEL_NAME&alias=$MODEL_ALIAS")

if [[ $FORCE_TRAIN == true || $alias_status != 200 ]]; then
  log "Обучение модели (около 2 минут)"
  docker compose run --rm train
else
  echo "В реестре уже есть $MODEL_NAME@$MODEL_ALIAS, обучение пропускаю (--train, чтобы обучить заново)"
fi

# сервис проверяет алиас раз в 30 с и загружает модель — /health отвечает 503, пока её нет
log "Ожидание сервиса с моделью"
wait_for API "$API_URL/health" 180

log "Готово"
echo "Версия модели: $(curl -fsS "$API_URL/health")"
cat <<EOF

  API (Swagger)   $API_URL/docs
  MLflow          $MLFLOW_URL
  Grafana         http://localhost:3000   (admin / admin)
  Prometheus      http://localhost:9090

  Логи:        docker compose logs -f api
  Остановить:  docker compose down
EOF
