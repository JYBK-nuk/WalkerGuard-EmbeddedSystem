<template>
  <head>
    <title>Camera Capture with Vue 3</title>
  </head>
  <body>
    <div id="app" style="display: flex">
      <button @click="connect()">Connect</button>
      <div>
        <video ref="videoElement" autoplay playsinline></video>
        <canvas ref="canvasElement" style="display: none"></canvas>
      </div>
      {{ str }}
      <div class="history">
        <div v-for="image in images.slice().reverse()">
          <img :src="image" />
        </div>
      </div>
    </div>
  </body>
</template>
<style>
img {
  height: 80px;
}
</style>
<script>
const app = Vue.createApp({
  data() {
    return {
      video: null,
      canvas: null,
      ws: null,
      images: [],
      str: "",
    };
  },
  mounted() {
    this.video = this.$refs.videoElement;
    this.canvas = this.$refs.canvasElement;

    // Get user media and display the video stream in the video element
    navigator.mediaDevices
      .getUserMedia({
        video: {
          width: { ideal: 1920 },
          height: { ideal: 1080 },
          facingMode: { exact: "environment" },
        },
        audio: false,
      })
      .then((stream) => {
        this.video.srcObject = stream;
        this.init();
      })
      .catch((error) => {
        console.error("Error accessing camera:", error);
      });

    setInterval(() => {
      this.updateImages();
    }, 100);
  },
  methods: {
    async init() {
      console.log(await navigator.mediaDevices.enumerateDevices());
      this.str = await navigator.mediaDevices.enumerateDevices();
    },
    connect() {
      if (this.ws) {
        this.ws.close();
      }
      this.ws = new WebSocket(`wss://${window.location.host}/event`);
      this.ws.onopen = () => {
        console.log("WebSocket connection established.");
      };

      this.ws.onmessage = (event) => {
        const message = JSON.parse(event.data);
        if (message.event === "get") {
          const images = this.images.slice();
          this.sendImages(images);
        }
      };
    },
    updateImages() {
      this.canvas.width = this.video.videoWidth;
      this.canvas.height = this.video.videoHeight;
      // Capture and store an image from the video stream
      this.canvas
        .getContext("2d")
        .drawImage(this.video, 0, 0, this.canvas.width, this.canvas.height);
      var image = this.canvas.toDataURL("image/jpeg");

      if (image == "data:,") return;

      this.images.push(image);

      if (this.images.length > 5) {
        this.images.shift();
      }
    },
    sendImages(images) {
      const formData = new FormData();
      for (let i = 0; i < images.length; i++) {
        const file = new File([images[i]], images[i].name, {
          type: images[i].type,
        });
        formData.append("images", file);
      }

      fetch(`${window.location.origin}/cam2`, {
        method: "PUT",
        body: formData,
      })
        .then((response) => {
          console.log("Images uploaded successfully");
        })
        .catch((error) => {
          console.error("Error uploading images:", error);
        });
    },
  },
});

app.mount("#app");
</script>
