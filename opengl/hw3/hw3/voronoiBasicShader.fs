#version 330 core
//out vec4 FragColor;

//in vec3 ourPosition;
uniform float inputPixel[32];

void main()
{
    // if(gl_FragCoord.x < 512){
    //     gl_FragColor = vec4(0,1,0,1.0);
    // }else{
    // gl_FragColor = vec4(0,0,1,1.0);
    // }
    int flag = 0;
    float jug = 512.0;
    int pos = 0;
    for(int i = 0;i < 32;++i){
        if(inputPixel[i]-jug>0.001||inputPixel[i]-jug<-0.001){
            flag = 1;
            pos = i;
            break;
        }
    } 
    int fjug = 1;
    if(flag == fjug){
        if(gl_FragCoord.x > (pos / 32.0) * 1024.0&&gl_FragCoord.x < ((pos+1) / 32.0) * 1024.0)
        gl_FragColor = vec4(0,0,1,1.0);
        else gl_FragColor = vec4(1,0,0,1.0);
    }else{
        gl_FragColor = vec4(0,1,0,1.0);
    }
    // int index = int(gl_FragCoord.x);
    // gl_FragColor = vec4(0,0,inputPixel[index]/1024,1.0);
}