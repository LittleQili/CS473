#version 330 core
//out vec4 FragColor;

//in vec3 ourPosition;

void main()
{
    gl_FragColor = vec4(gl_FragCoord.x/1024,gl_FragCoord.y/1024,0,1.0);
}