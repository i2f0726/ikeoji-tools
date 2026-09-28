# イケオジ計画のツール

https://tools.ikeojikeikaku.com/ で公開する、イケオジ計画（https://www.ikeojikeikaku.com/）のセルフチェック・診断ツール。GitHub Pages（main ブランチのルート）で配信する。

## face/ 顔の形スキャン

- スマホのインカメラ（または写真）から、MediaPipe Face Landmarker で顔の輪郭の点を取り、縦横の比率から「卵型・丸顔・面長・ベース型・逆三角」の近さを出す
- 読み取りはすべてブラウザ内。画像・数値は送信も保存もしない
- カメラが使えない人向けに、質問3つで調べるモードもある
- 判定の基準値は `face.js` の `AVG` / `PROTO` / `SD`。`?debug=1` を付けると、測った数値が画面に出るので、実際の顔で調整するときに使う
- 髪型の記事へのリンクは `LINKS` の `from`（公開日時）を過ぎたものだけ表示される

## ローカルで確認

```
npx http-server . -p 8765 -c-1
```

カメラは https か localhost でしか動かない。
