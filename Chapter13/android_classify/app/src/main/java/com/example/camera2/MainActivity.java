package com.example.camera2;

import androidx.appcompat.app.AppCompatActivity;

import android.content.Intent;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.Bundle;
import android.provider.MediaStore;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

import java.io.IOException;
import java.io.InputStream;
import java.util.Objects;

public class MainActivity extends AppCompatActivity {
    private ImageView imgCapture;
    private static final int Image_Capture_Code = 1;

    // Used to load the 'native-lib' library on application startup.
    static {
        System.loadLibrary("native-lib");
    }

    /**
     * A native methods that are implemented by the 'native-lib' native library,
     * which is packaged with this application.
     */
    public native String classifyBitmap(int[] pixels, int width, int height);
    public native void initClassifier(AssetManager assetManager);
    public native void destroyClassifier();

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        initClassifier(getAssets());

        setContentView(R.layout.activity_main);
        Button btnCapture = findViewById(R.id.btnTakePicture);
        imgCapture = findViewById(R.id.capturedImage);
        btnCapture.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Intent cInt = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
                startActivityForResult(cInt, Image_Capture_Code);
            }
        });
    }

    @Override
    protected void onDestroy(){
        super.onDestroy();
        destroyClassifier();
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        if (requestCode == Image_Capture_Code) {
            if (resultCode == RESULT_OK) {
                Bitmap bp = (Bitmap) Objects.requireNonNull(data.getExtras()).get("data");

//                InputStream istr = null;
//                try {
//                    istr = getAssets().open("dog.png");
//                } catch (IOException e) {
//                    e.printStackTrace();
//                }
//                Bitmap bp = BitmapFactory.decodeStream(istr);

                if (bp != null) {
                    Bitmap argb_bp = bp.copy(Bitmap.Config.ARGB_8888, true);
                    if (argb_bp != null) {
                        float ratio_w = (float) bp.getWidth() / (float) bp.getHeight();
                        float ratio_h = (float) bp.getHeight() / (float) bp.getWidth();

                        int width = 224;
                        int height = 224;

                        int new_width = Math.max((int) (height * ratio_w), width);
                        int new_height = Math.max(height, (int) (width * ratio_h));

                        Bitmap resized_bitmap = Bitmap.createScaledBitmap(
                                argb_bp, new_width, new_height, false);
                        Bitmap cropped_bitmap = Bitmap.createBitmap(resized_bitmap, 0, 0, width, height);


                        int[] pixels = new int[width * height];
                        cropped_bitmap.getPixels(pixels, 0, width, 0, 0, width, height);
                        String class_name = classifyBitmap(pixels, width, height);


                        cropped_bitmap.setPixels(pixels, 0, width, 0 ,0, width, height);
                        imgCapture.setImageBitmap(cropped_bitmap);

                        TextView class_view = findViewById(R.id.textViewClass);
                        class_view.setText(class_name);
                    }
                }

            } else if (resultCode == RESULT_CANCELED) {
                Toast.makeText(this, "Cancelled", Toast.LENGTH_LONG).show();
            }
        }
    }
}
