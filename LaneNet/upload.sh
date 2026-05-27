#!/usr/bin/env zsh
set -euo pipefail

NAMESPACE="${NAMESPACE:-ucm-vista}"
POD="${POD:-lanenet-data-uploader}"
REMOTE_ROOT="${REMOTE_ROOT:-/data}"
LOCAL_ROOT="${LOCAL_ROOT:-$(cd -- "$(dirname -- "$0")" && pwd)}"

log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

human_size() {
  du -sh "$1" | awk '{print $1}'
}

byte_size() {
  du -sk "$1" | awk '{print $1 * 1024}'
}

elapsed_seconds() {
  local start="$1"
  local now
  now="$(date +%s)"
  echo $((now - start))
}

log "Local root:  $LOCAL_ROOT"
log "Remote pod:  $NAMESPACE/$POD"
log "Remote root: $REMOTE_ROOT"

cd "$LOCAL_ROOT"

items=(archive/*(N) CuLane/*(N))

if (( ${#items[@]} == 0 )); then
  log "No files or directories found under archive/ or CuLane/"
  exit 1
fi

log "Found ${#items[@]} upload sections."

section_number=0
for item in "${items[@]}"; do
  ((++section_number))

  parent="$(dirname -- "$item")"
  name="$(basename -- "$item")"
  remote_dir="$REMOTE_ROOT/$parent"
  size_human="$(human_size "$item")"
  size_bytes="$(byte_size "$item")"
  started_at="$(date +%s)"

  log "--------------------------------------------------------------------------------"
  log "Section $section_number/${#items[@]}"
  log "Uploading item: $item"
  log "Item size:      $size_human"
  log "Remote target:  $remote_dir/$name"
  log "Creating remote directory: $remote_dir"

  kubectl exec -n "$NAMESPACE" "$POD" -- mkdir -p "$remote_dir"

  log "Starting tar stream. Each path printed by tar below is being uploaded."

  if command -v pv >/dev/null 2>&1; then
    tar -C "$parent" -cvf - "$name" \
      | pv -s "$size_bytes" -pterab \
      | kubectl exec -i "$POD" -n "$NAMESPACE" -- tar -C "$remote_dir" -xf -
  else
    log "pv not found; using dd status=progress for byte count and speed."
    log "Install pv for nicer progress: brew install pv"
    tar -C "$parent" -cvf - "$name" \
      | dd bs=4m status=progress \
      | kubectl exec -i "$POD" -n "$NAMESPACE" -- tar -C "$remote_dir" -xf -
  fi

  elapsed="$(elapsed_seconds "$started_at")"
  log "Completed upload: $item"
  log "Elapsed seconds:  $elapsed"
  log "Remote size now:"
  kubectl exec -n "$NAMESPACE" "$POD" -- du -sh "$remote_dir/$name" || true
done

log "--------------------------------------------------------------------------------"
log "All sections completed."
log "Final remote usage:"
kubectl exec -n "$NAMESPACE" "$POD" -- du -sh "$REMOTE_ROOT/archive" "$REMOTE_ROOT/CuLane" || true
