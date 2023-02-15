import { Container, Flex, Spacer, useToast } from "@chakra-ui/react";
import Footer from "./footer";
// import { useAuthState } from "react-firebase-hooks/auth";
// import LoginModal from "../components/loginModal";
import Navbar from "./navbar";
// import { auth, logOut } from "../utils/firebase/clientApp";

interface LayoutProps {
  children: React.ReactNode;
}

const Layout = ({ children }: LayoutProps) => {
  const toast = useToast();

  // const {
  //   isOpen: isOpenLoginModal,
  //   onClose: onCloseLoginModal,
  //   onOpen: onOpenLoginModal,
  // } = useDisclosure();

  // const [user, loading, error] = useAuthState(auth!);

  // useEffect(() => {
  //   console.log('user', user);
  // }, [user]);

  return (
    <Flex
      h="100vh"
      flexDirection="column"
      color="gray.600"
      // using 100vw inside this flex will cause the page to
      // have scroll-x, this will prevent that
      overflowX="hidden"
    >
      {/* <LoginModal isOpen={isOpenLoginModal} onClose={onCloseLoginModal} /> */}
      <Navbar
        // onClickLogin={onOpenLoginModal}
        // onClickLogout={logOut}
        // isLoadingUserStatus={loading}
        // isLoggedIn={!!user}
        onClickLogin={() => {
          toast({
            variant: "left-accent",
            description: "ระบบลงทะเบียนยังไม่เปิดใช้งานในเวลานี้",
            status: "warning",
            duration: 3000,
            isClosable: true,
            position: "top-right",
          });
        }}
        onClickLogout={() => {}}
        isLoadingUserStatus={false}
        isLoggedIn={false}
      />
      <Container maxW="container.lg">{children}</Container>
      <Spacer />
      <Footer />
    </Flex>
  );
};

export default Layout;
