#ifndef _BCAST_H_
#define _BCAST_H_

#include <stdexcept>
#include <stdio.h>
#include <string.h>
#include <sys/types.h>
#include <linux/in.h>
#include <sys/socket.h>
#include <sys/select.h>

namespace udp_bcast {

#define IP_FOUND "IP_FOUND"
#define IP_FOUND_ACK "IP_FOUND_ACK"

    class udp_bcast_runtime_error: public std::runtime_error {
        public:
            udp_bcast_runtime_error(const char *w) : std::runtime_error(w) {}
    };

    class server {
        public:
            server(int port);
            //~server();
            int recv(char *msg, size_t max_size);
            int timed_recv(char *msg, size_t max_size, int max_wait_ms);
        private:
            int port;
            int sock;
            int yes = 1;
            struct sockaddr_in client_addr;
            struct sockaddr_in server_addr;
            int addr_len;
            int count;
            int ret;
            fd_set readfd;
    };

    class client {
        public:
            client(int port);
            //~client();
            int send(const char *msg, size_t size);
        private:
            int port;
            int sock;
            int yes = 1;
            struct sockaddr_in broadcast_addr;
            struct sockaddr_in server_addr;
            int addr_len;
            int count;
            int ret;
            fd_set readfd;
            int i;
    };

} // namespace udp_bcast

#endif