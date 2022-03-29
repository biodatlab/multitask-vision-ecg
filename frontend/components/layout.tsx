import { Container, Flex, Spacer, useDisclosure } from "@chakra-ui/react";
import { useEffect } from "react";
// import { useAuthState } from "react-firebase-hooks/auth";
// import LoginModal from "../components/loginModal";
import Navbar from "../components/navbar";
import Footer from "../components/footer";
// import { auth, logOut } from "../utils/firebase/clientApp";

interface LayoutProps {
  children: React.ReactNode;
}

const Layout = ({ children }: LayoutProps) => {
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
      background="#fefefd"
      color="gray.600"
    >
      {/* <LoginModal isOpen={isOpenLoginModal} onClose={onCloseLoginModal} /> */}
      <Navbar
        // onClickLogin={onOpenLoginModal}
        // onClickLogout={logOut}
        // isLoadingUserStatus={loading}
        // isLoggedIn={!!user}
        onClickLogin={() => {}}
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
