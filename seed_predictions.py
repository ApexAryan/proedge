import urllib.request, json

for i in range(1, 21):
    payload = json.dumps({
        "sport": "nba",
        "home_team": "BOS",
        "away_team": "LAL",
        "game_date": "2026-04-30T19:00:00",
        "total_line": 220 + i,
        "home_rest_days": i % 4,
        "away_rest_days": i % 3
    }).encode()
    req = urllib.request.Request(
        "http://localhost:8080/predictions",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST"
    )
    urllib.request.urlopen(req)
    print(f"Sent {i}/20")

print("Done")
