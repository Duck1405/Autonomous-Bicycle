#!/usr/bin/env bash
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

line_count() {
  wc -l "$1" | awk '{print $1}'
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

items=(
  "/home/aman/Projects/Auto/100k"
  "/home/aman/Projects/Auto/100k_images"
  "/home/aman/Projects/Auto/ExtraHybridNetGithubData/bdd_seg_gt"
  "/home/aman/Projects/Auto/ExtraHybridNetGithubData/bdd_lane_gt"
)

if (( ${#items[@]} == 0 )); then
  log "No files or directories configured for upload."
  exit 1
fi

log "Found ${#items[@]} upload sections."

section_number=0
remote_targets=()
for item in "${items[@]}"; do
  ((++section_number))

  if [[ ! -e "$item" ]]; then
    log "Missing local path: $item"
    exit 1
  fi

  parent="$(dirname -- "$item")"
  name="$(basename -- "$item")"
  remote_dir="$REMOTE_ROOT"
  remote_targets+=("$remote_dir/$name")
  size_human="$(human_size "$item")"
  started_at="$(date +%s)"
  local_list="$(mktemp /tmp/upload-local.XXXXXX)"
  remote_list="$(mktemp /tmp/upload-remote.XXXXXX)"
  missing_list="$(mktemp /tmp/upload-missing.XXXXXX)"

  log "--------------------------------------------------------------------------------"
  log "Section $section_number/${#items[@]}"
  log "Uploading item: $item"
  log "Item size:      $size_human"
  log "Remote target:  $remote_dir/$name"
  log "Creating remote directory: $remote_dir"

  kubectl exec -n "$NAMESPACE" "$POD" -- mkdir -p "$remote_dir"

  log "Checking which files are missing on the remote pod."

  (
    cd "$parent"
    find "$name" -type f | sort
  ) > "$local_list"

  kubectl exec -n "$NAMESPACE" "$POD" -- sh -c \
    'cd "$1" && if [ -e "$2" ]; then find "$2" -type f; fi' \
    sh "$remote_dir" "$name" | sort > "$remote_list"

  comm -23 "$local_list" "$remote_list" > "$missing_list"

  total_count="$(line_count "$local_list")"
  missing_count="$(line_count "$missing_list")"

  log "Local files:      $total_count"
  log "Missing remotely: $missing_count"

  if (( missing_count == 0 )); then
    log "Skipping upload because all files already exist remotely."
    rm -f "$local_list" "$remote_list" "$missing_list"
    continue
  fi

  log "Starting tar stream. Each path printed by tar below is being uploaded."
  log "Already-present remote files are not included in this tar stream."

  if command -v pv >/dev/null 2>&1; then
    tar -C "$parent" -cvf - -T "$missing_list" \
      | pv -pterab \
      | kubectl exec -i "$POD" -n "$NAMESPACE" -- tar -C "$remote_dir" -xf -
  else
    log "pv not found; using dd status=progress for byte count and speed."
    log "Install pv for nicer progress: brew install pv"
    tar -C "$parent" -cvf - -T "$missing_list" \
      | dd bs=4M status=progress \
      | kubectl exec -i "$POD" -n "$NAMESPACE" -- tar -C "$remote_dir" -xf -
  fi

  rm -f "$local_list" "$remote_list" "$missing_list"

  elapsed="$(elapsed_seconds "$started_at")"
  log "Completed upload: $item"
  log "Elapsed seconds:  $elapsed"
  log "Remote size now:"
  kubectl exec -n "$NAMESPACE" "$POD" -- du -sh "$remote_dir/$name" || true
done

log "--------------------------------------------------------------------------------"
log "All sections completed."
log "Final remote usage:"
kubectl exec -n "$NAMESPACE" "$POD" -- du -sh "${remote_targets[@]}" || true
