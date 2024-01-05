package com.example.demo;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.IOException;
import java.net.MalformedURLException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.util.HashMap;
import java.util.List;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.core.io.ClassPathResource;
import org.springframework.core.io.Resource;
import org.springframework.core.io.UrlResource;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.CrossOrigin;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.multipart.MultipartFile;

import com.fasterxml.jackson.core.exc.StreamReadException;
import com.fasterxml.jackson.databind.DatabindException;
import com.fasterxml.jackson.databind.ObjectMapper;

import org.springframework.http.HttpHeaders;
import org.springframework.http.HttpStatus;
import org.springframework.http.MediaType;
import org.springframework.web.bind.annotation.PutMapping;
import org.springframework.web.bind.annotation.PathVariable;
import java.util.zip.ZipEntry;
import java.util.zip.ZipOutputStream;

class Point {
	public int x;
	public int y;

	public String toString() {
		return "x: " + x + ", y: " + y;
	}
}

class Record {
	public String plate;
	public String img1;
	public String img2;
	public String time;

}

@SpringBootApplication
public class DemoApplication {

	public static void main(String[] args) {
		SpringApplication.run(DemoApplication.class, args);
	}

	@RestController
	public class DemoController {
		HashMap<String, File> images = new HashMap<String, File>();

		// 取得傳來的圖片
		@PostMapping("/image/{id}")
		public String handleImageUpload(@PathVariable String id, @RequestParam("image") MultipartFile image) {
			try {
				if (!image.isEmpty()) {
					String filePath = ".\\src\\main\\resources\\static\\" + id + ".jpg";
					Path destination = new File(filePath).toPath();

					Files.copy(image.getInputStream(), destination, StandardCopyOption.REPLACE_EXISTING);
					// 將檔案保存到指定路徑

					images.put(id, destination.toFile());
					return "Image uploaded successfully.";
				} else {
					return "No image uploaded.";
				}
			} catch (IOException e) {
				return "Error uploading image.";
			}
		}

		// 從資料夾取得圖片
		@GetMapping("/image/{id}")
		public ResponseEntity<Resource> getImage(@PathVariable String id) {
			// 根據ID檢索圖片檔案
			File file = images.get(id);// new File(".\\src\\main\\resources\\static\\" + id + ".jpg");
			if (file != null && file.exists()) {
				try {
					// 建立 Resource 物件並回傳圖片檔案
					Resource resource = new UrlResource(file.toURI());
					return ResponseEntity.ok()
							.contentType(MediaType.IMAGE_JPEG)
							.body(resource);
				} catch (MalformedURLException e) {
					return ResponseEntity.notFound().build();
				}
			} else {
				return ResponseEntity.notFound().build();
			}
		}

		// 接收點完的座標
		@PostMapping("/points")
		public ResponseEntity<?> getPoints(@RequestBody List<Point> points) {
			for (Point point : points) {
				System.out.println(point.toString());
			}
			return ResponseEntity.ok(null);
		}

		// 取得違規紀錄
		@PostMapping("/record")
		// 印出當前路徑
		File recordFile = new File(
				"deliver_photo\\demo\\src\\main\\resources\\static\\record\\1.json");

		File file1;
		File file2;
		// 讀取JSON檔案

		ObjectMapper mapper = new ObjectMapper();
		Record record = mapper.readValue(recordFile, Record.class);

	}

}