#include "udp_bcast.h"

namespace udp_bcast {

    server::server(int port): port(port) {
        sock = socket(AF_INET, SOCK_DGRAM, 0);
        if (sock < 0) {
            throw udp_bcast_runtime_error(("sock error").c_str());
        }

        addr_len = sizeof(struct sockaddr_in);

        memset((void*)&server_addr, 0, addr_len);
        server_addr.sin_family = AF_INET;
        server_addr.sin_addr.s_addr = htons(INADDR_ANY);
        server_addr.sin_port = htons(port);

        ret = bind(sock, (struct sockaddr*)&server_addr, addr_len);
        if (ret < 0) {
            throw udp_bcast_runtime_error(("bind error").c_str());
        }
    }

    int server::recv(char *msg, size_t max_size) {
        FD_ZERO(&readfd);
        FD_SET(sock, &readfd);

        ret = select(sock+1, &readfd, NULL, NULL, 0);
        if(ret == -1)
        {
            return -1;
        }
        if (ret > 0) {
            if (FD_ISSET(sock, &readfd)) {
                return recvfrom(sock, msg, max_size, 0, (struct sockaddr*)&client_addr, &addr_len);
            }
        }

        errno = EAGAIN;
        return -1;
    }

    int server::timed_recv(char *msg, size_t max_size, int max_wait_ms) {
        FD_ZERO(&readfd);
        FD_SET(sock, &readfd);


        struct timeval timeout;
        timeout.tv_sec = max_wait_ms / 1000;
        timeout.tv_usec = (max_wait_ms % 1000) * 1000;

        ret = select(sock+1, &readfd, &readfd, NU&readfdLL, &timeout);
        if(ret == -1)
        {
            return -1;
        }
        if (ret > 0) {
            if (FD_ISSET(sock, &readfd)) {
                return recvfrom(sock, msg, max_size, 0, (struct sockaddr*)&client_addr, &addr_len);
            }
        }

        errno = EAGAIN;
        return -1;
    }

    client::client(int port): port(port) {
        sock = socket(AF_INET, SOCK_DGRAM, 0);
        if (sock < 0) {
            throw udp_bcast_runtime_error(("sock error").c_str());
        }
        ret = setsockopt(sock, SOL_SOCKET, SO_BROADCAST, (char*)&yes, sizeof(yes));
        if (ret == -1) {
            throw udp_bcast_runtime_error(("bind error").c_str());
        }

        addr_len = sizeof(struct sockaddr_in);

        memset((void*)&broadcast_addr, 0, addr_len);
        broadcast_addr.sin_family = AF_INET;
        broadcast_addr.sin_addr.s_addr = htonl(INADDR_BROADCAST);
        broadcast_addr.sin_port = htons(port);

    }

    client::send(const char *msg, size_t size) {
        return sendto(sock, msg, size, 0, (struct sockaddr*) &broadcast_addr, addr_len);
    }


} // namespace udp_bcast