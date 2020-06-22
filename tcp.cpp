#include "tcp.h"

#include <stdio.h> 
#include <stdlib.h> 
#include <unistd.h> 
#include <string.h> 
#include <sys/types.h> 
#include <sys/socket.h> 
#include <arpa/inet.h> 
#include <netinet/in.h> 
#include <errno.h>

namespace tcp {

    tcpclient::tcpclient(int port): mPort(port) {

        // Creating socket file descriptor 
        if ( (sockfd = socket(AF_INET, SOCK_STREAM, 0)) < 0 ) { 
            throw "error sock";
        }
         
        // Filling server information 
        servaddr.sin_family = AF_INET; 
        servaddr.sin_port = htons(mPort); 
        //servaddr.sin_addr.s_addr = htonl(INADDR_BROADCAST);  

        if (inet_pton(AF_INET,"127.0.0.1",&servaddr.sin_addr)<=0) {
            throw "error sin_addr";
        }

        if (connect(sockfd,(struct sockaddr *)&servaddr,sizeof(servaddr))<0) {
            throw "error connect";
        }

        // int on=1;
        // setsockopt(sockfd, SOL_SOCKET, SO_BROADCAST, &on, sizeof(on));
        
        memset(&servaddr, 0, sizeof(servaddr)); 
              
    }
    
    tcpclient::~tcpclient() {
        close(sockfd);
    }
    
    int tcpclient::sendit(const char* message, size_t length) {
        char* buff = new char[length];
        send(sockfd, message, length, 0); 
        read(sockfd, buff, length);
        return 0;
    }
    
    tcpserver::tcpserver(int port): mPort(port) {
            
        int opt=1;
        int new_socket;

        // Creating socket file descriptor 
        if ( (sockfd = socket(AF_INET, SOCK_STREAM, 0)) < 0 ) { 
            throw "error sock";
        } 

        if (setsockopt(sockfd,SOL_SOCKET,SO_REUSEADDR|SO_REUSEPORT,&opt,sizeof(opt))) {
            throw "error sock opt";
        }
          
        memset(&servaddr, 0, sizeof(servaddr)); 
        memset(&cliaddr, 0, sizeof(cliaddr)); 
          
        // Filling server information 
        servaddr.sin_family    = AF_INET; // IPv4 
        servaddr.sin_addr.s_addr = htons(INADDR_ANY); 
        servaddr.sin_port = htons(mPort); 

        int addrlen = sizeof(servaddr);
          
        // Bind the socket with the server address 
        if ( bind(sockfd, (const struct sockaddr *)&servaddr, sizeof(servaddr)) < 0 ) { 
            throw "error bind";
        } 

        if (listen(sockfd,3)<0) {
            throw "error listen";
        }

        if ((new_socket = accept(sockfd,(struct sockaddr *)&servaddr,
            (socklen_t*)&addrlen))<0) {
            throw "error accept";
        }
    }
    
    tcpserver::~tcpserver() {
        close(sockfd);
    }
    
    int tcpserver::recv(char* buffer, size_t length) {
        unsigned int n; 
        n = read(sockfd, buffer, length); 
        send(sockfd,buffer,length,0);
        return n;
    }
}
